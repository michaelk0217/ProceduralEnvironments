#include "Engine.h"

void Engine::run()
{
	initGlfwWindow();
	initVulkan();
	mainLoop();
	cleanUp();
}

void Engine::initGlfwWindow()
{
    try {
        window = std::make_unique<Window>(windowConfig.width, windowConfig.height, windowConfig.windowTitle);

        window->setAppFramebufferResizeCallback([this](int width, int height)
            {
                this->windowConfig.resized = true;
            }
        );

    }
    catch (const std::runtime_error& e)
    {
        std::cerr << "Window Initialization Error: " << e.what() << std::endl;
        throw;
    }
}

void Engine::initVulkan()
{

    slang::createGlobalSession(&slangGlobalSession);
    
    device = std::make_unique<VulkanDevice>(window->getGlfwWindow());
    swapchain = std::make_unique<VulkanSwapchain>(device->instance, device->surface, device->logicalDevice, device->physicalDevice, window->getGlfwWindow());
    swapchain->create(windowConfig.width, windowConfig.height);

    frameCommandBuffers = VulkanDevice::createCommandBuffers(device->logicalDevice, VK_COMMAND_BUFFER_LEVEL_PRIMARY, device->graphicsCommandPool, MAX_CONCURRENT_FRAMES);

    createDescriptorPool();
    
    createHeightMapResources(heightMapSize, VK_FORMAT_R32_SFLOAT);
    initializeVertexIndexBuffers();
    createTerrainGenerationComputeResources();

    heightMapConfig.seed = 12345;
    heightMapConfig.offset[0] = 0.0f;
    heightMapConfig.offset[1] = 0.0f;
    heightMapConfig.frequency = 0.01f;
    heightMapConfig.octaves = 8;
    heightMapConfig.lacunarity = 2.0f;
    heightMapConfig.persistence = 0.5f;
    heightMapConfig.noiseScale = 1.0f;

    terrainGenParams.gridResolution = static_cast<uint32_t>(heightMapSize);
    terrainGenParams.heightScale = 3.0f;
    terrainGenParams.normalsStrength = 50.0f;
    terrainGenParams.terrainSideLength = 40.0f;
    

    createGraphicsResources();

    createDepthResources();

    createSyncPrimitives();

    

    camera = std::make_unique<Camera>(
        glm::vec3(0.0f, 2.0, 2.0), // pos
        glm::vec3(0.0f, 1.0f, 0.0f), // worldupvector: set to y-up
        -90.0f, // yaw : look along x axis
        -50.0f, // pitch
        3.0f, // movementSpeed
        0.1f, // turnspeed
        60.0f, // fov
        (float)windowConfig.width / (float)windowConfig.height,
        0.1f,
        500.0f
    );

    uiOverlay = std::make_unique<UIOverlay>(*window, *device, *swapchain);


}

void Engine::mainLoop()
{
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (window && !window->shouldClose())
    {
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
        lastTime = currentTime;
        totalElapsedTime += deltaTime;

        UIOverlay::update_frame_history(frame_history, 1.0f / deltaTime);

        window->pollEvents();
        processInput(deltaTime);

        glm::vec3 cameraDir = camera->getCameraDirection();

        UIPacket uiPacket{
            deltaTime,
            totalElapsedTime,
            frame_history,
            cameraDir,
            heightMapConfig,
            heightMapConfigChanged,
            terrainGenParams
        };
        uiOverlay->newFrame();
        uiOverlay->buildUI(uiPacket);
        

        
        drawFrame();
    }
}

void Engine::cleanUp()
{
    vkDeviceWaitIdle(device->logicalDevice);

    uiOverlay.reset();

    cleanUpTerrainGenerationComputeResources();
    cleanUpHeightMapResources();

    //debugPrintIndexBuffer();

    cleanUpVertexIndexBuffers();

    cleanUpSyncPrimitives();

    depthStencil.destroy();

    cleanUpGraphicsResources();

    vkDestroyDescriptorPool(device->logicalDevice, descriptorPool, nullptr);

    swapchain.reset();
    device.reset();
    camera.reset();
    window.reset();

    std::cout << generationCalls << " times generated" << std::endl;
}

void Engine::drawFrame()
{
    vkWaitForFences(device->logicalDevice, 1, &waitFences[currentFrame], VK_TRUE, UINT64_MAX);

    VK_CHECK_RESULT(vkResetFences(device->logicalDevice, 1, &waitFences[currentFrame]));

    uint32_t imageIndex{ 0 };
    VkResult result = swapchain->acquireNextImage(presentCompleteSemaphores[currentFrame], imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        windowResize();
        return;
    }
    else if ((result != VK_SUCCESS) && (result != VK_SUBOPTIMAL_KHR))
    {
        throw "Could not acquire the next swapchain image";
    }

    MVPMatrices mvpData{};
    mvpData.model = glm::mat4(1.0f);
    mvpData.view = camera->calculateViewMatrix();
    mvpData.proj = camera->getProjectionMatrix();
    mvpData.mvp = mvpData.proj * mvpData.view * mvpData.model;
    mvpData.modelInverse = glm::inverse(mvpData.model);
    mvpData.viewInverse = glm::inverse(mvpData.view);
    mvpData.projInverse = glm::inverse(mvpData.proj);

    graphicsUBO[currentFrame].copyTo(&mvpData, sizeof(MVPMatrices));

    vkResetCommandBuffer(frameCommandBuffers[currentFrame], 0);
    VkCommandBufferBeginInfo cmdBufInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    const VkCommandBuffer commandBuffer = frameCommandBuffers[currentFrame];
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));

    if (!heightMapInitialized || heightMapConfigChanged)
    {
        //recordHeightMapGeneration(commandBuffer, heightMapConfig, heightMapSize);
        recordTerrainMeshGeneration(commandBuffer, heightMapConfig, terrainGenParams);
    }

    vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        swapchain->images[imageIndex],
        0,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        depthStencil.image,
        0,
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1 }
    );

    VkRenderingAttachmentInfo colorAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    colorAttachment.imageView = swapchain->imageViews[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = { 0.02f, 0.02f, 0.02f, 0.0f };

    VkRenderingAttachmentInfo depthStencilAttachment{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    depthStencilAttachment.imageView = depthStencil.imageView;
    depthStencilAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthStencilAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthStencilAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthStencilAttachment.clearValue.depthStencil = { 1.0f, 0 };

    VkRenderingInfo renderingInfo{ VK_STRUCTURE_TYPE_RENDERING_INFO_KHR };
    renderingInfo.renderArea = { 0, 0, windowConfig.width, windowConfig.height };
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthStencilAttachment;
    renderingInfo.pStencilAttachment = &depthStencilAttachment;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    //VkViewport viewport{ 0.0f, 0.0f, (float)windowConfig.width, (float)windowConfig.height, 0.0f, 1.0f };
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = (float)windowConfig.height;
    viewport.width = (float)windowConfig.width;
    viewport.height = -(float)windowConfig.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{ 0, 0, windowConfig.width, windowConfig.height };
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        graphicsPipelineLayout,
        0, 1,
        &graphicsDescriptors[currentFrame],
        0, nullptr
    );

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

    //vkCmdPushConstants(commandBuffer, graphicsPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VertexShaderPushConstant), &vertPushConstant);

    VkDeviceSize offsets[1]{ 0 };

    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.buffer, offsets);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    //vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(quadsIndices.size()), 1, 0, 0, 0);
    uint32_t indexCount = (heightMapSize - 1) * (heightMapSize - 1) * 6;
    vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);

    vkCmdEndRendering(commandBuffer);


    uiOverlay->render(commandBuffer, imageIndex, windowConfig.width, windowConfig.height);

    vks::tools::insertImageMemoryBarrier(
        commandBuffer,
        swapchain->images[imageIndex],
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        0,
        VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

    VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pWaitDstStageMask = &waitStageMask;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.commandBufferCount = 1;
    submitInfo.pWaitSemaphores = &presentCompleteSemaphores[currentFrame];
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderCompleteSemaphores[imageIndex];
    submitInfo.signalSemaphoreCount = 1;

    VK_CHECK_RESULT(vkQueueSubmit(device->graphicsQueue, 1, &submitInfo, waitFences[currentFrame]));

    result = swapchain->queuePresent(device->presentQueue, imageIndex, renderCompleteSemaphores[imageIndex]);

    if ((result == VK_ERROR_OUT_OF_DATE_KHR) || (result == VK_SUBOPTIMAL_KHR))
    {
        windowResize();
    }
    else if (result != VK_SUCCESS)
    {
        throw std::runtime_error("Could not present the image to the swap chain");
    }

    currentFrame = (currentFrame + 1) % MAX_CONCURRENT_FRAMES;
}

void Engine::windowResize()
{
    int newWidth = 0, newHeight = 0;
    window->getFramebufferSize(newWidth, newHeight);
    while (newWidth == 0 || newHeight == 0)
    {
        glfwWaitEvents();
        window->getFramebufferSize(newWidth, newHeight);
        // todo: implement window->waitEvents() wait if window is minimized
    }
    vkDeviceWaitIdle(device->logicalDevice);
    this->windowConfig.width = newWidth;
    this->windowConfig.height = newHeight;

    // depthResource cleanup
    depthStencil.destroy();

    swapchain.reset();
    uiOverlay.reset();

    swapchain = std::make_unique<VulkanSwapchain>(device->instance, device->surface, device->logicalDevice, device->physicalDevice, window->getGlfwWindow());
    swapchain->create(this->windowConfig.width, this->windowConfig.height);


    // depthResource recreate
    createDepthResources();
    uiOverlay = std::make_unique<UIOverlay>(*window, *device, *swapchain);
    camera->setAspectRatio((float)windowConfig.width / (float)windowConfig.height);
}

void Engine::processInput(float deltaTime)
{
    if (window->isMouseButtonPressed(GLFW_MOUSE_BUTTON_MIDDLE))
    {
        camera->processKeyboard(window->getKeys(), deltaTime);
        glfwSetInputMode(window->getGlfwWindow(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        camera->processMouseMovement(window->getXChange(), window->getYChange(), true);
    }
    else
    {
        glfwSetInputMode(window->getGlfwWindow(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
}

void Engine::createSyncPrimitives()
{
    for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++)
    {
        VkFenceCreateInfo fenceCI{};
        fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VK_CHECK_RESULT(vkCreateFence(device->logicalDevice, &fenceCI, nullptr, &waitFences[i]));
    }

    presentCompleteSemaphores.resize(MAX_CONCURRENT_FRAMES);
    for (auto& semaphore : presentCompleteSemaphores)
    {
        VkSemaphoreCreateInfo semaphoreCI{};
        semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK_RESULT(vkCreateSemaphore(device->logicalDevice, &semaphoreCI, nullptr, &semaphore));
    }

    renderCompleteSemaphores.resize(swapchain->images.size());
    for (auto& semaphore : renderCompleteSemaphores)
    {
        VkSemaphoreCreateInfo semaphoreCI{};
        semaphoreCI.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        VK_CHECK_RESULT(vkCreateSemaphore(device->logicalDevice, &semaphoreCI, nullptr, &semaphore));
    }
}

void Engine::cleanUpSyncPrimitives()
{
    for (size_t i = 0; i < presentCompleteSemaphores.size(); i++) vkDestroySemaphore(device->logicalDevice, presentCompleteSemaphores[i], nullptr);
    for (size_t i = 0; i < renderCompleteSemaphores.size(); i++) vkDestroySemaphore(device->logicalDevice, renderCompleteSemaphores[i], nullptr);
    for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++) vkDestroyFence(device->logicalDevice, waitFences[i], nullptr);
}

void Engine::createDescriptorPool()
{
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MAX_CONCURRENT_FRAMES + 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2},
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_CONCURRENT_FRAMES * 2 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}
    };

    VkDescriptorPoolCreateInfo poolCI{};
    poolCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCI.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCI.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolCI.pPoolSizes = poolSizes.data();
    poolCI.maxSets = MAX_CONCURRENT_FRAMES + 6;

    VK_CHECK_RESULT(vkCreateDescriptorPool(device->logicalDevice, &poolCI, nullptr, &descriptorPool));
}

void Engine::createGraphicsResources()
{
    // initialize graphics ubo
    graphicsUBO.resize(MAX_CONCURRENT_FRAMES);

    for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++)
    {
        graphicsUBO[i].create(
            device->logicalDevice,
            device->physicalDevice,
            sizeof(MVPMatrices),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );
        graphicsUBO[i].map();
    }

    // descriptor set layout
    std::array<VkDescriptorSetLayoutBinding, 1> bindings{};
    // mvp ubo
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    //// height map
    //bindings[1].binding = 1;
    //bindings[1].descriptorCount = 1;
    //bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    //bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    //// normal map
    //bindings[2].binding = 2;
    //bindings[2].descriptorCount = 1;
    //bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    //bindings[2].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorLayoutCI{};
    descriptorLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorLayoutCI.bindingCount = static_cast<uint32_t>(bindings.size());
    descriptorLayoutCI.pBindings = bindings.data();

    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device->logicalDevice, &descriptorLayoutCI, nullptr, &graphicsDescriptorSetLayout));

    //VkPushConstantRange pushConstant{};
    //pushConstant.offset = 0;
    //pushConstant.size = sizeof(VertexShaderPushConstant);
    //pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    // pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutCI{};
    pipelineLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCI.setLayoutCount = 1;
    pipelineLayoutCI.pSetLayouts = &graphicsDescriptorSetLayout;
    //pipelineLayoutCI.pushConstantRangeCount = 1;
    //pipelineLayoutCI.pPushConstantRanges = &pushConstant;

    VK_CHECK_RESULT(vkCreatePipelineLayout(device->logicalDevice, &pipelineLayoutCI, nullptr, &graphicsPipelineLayout));

    //VkShaderModule vertShaderModule = vks::tools::loadSlangShader(device->logicalDevice, slangGlobalSession, "shaders/shader.slang", "vertexMain");
    //VkShaderModule fragShaderModule = vks::tools::loadSlangShader(device->logicalDevice, slangGlobalSession, "shaders/shader.slang", "fragmentMain");
    VkShaderModule vertShaderModule = vks::tools::loadShader("shaders/vert.spirv", device->logicalDevice);
    VkShaderModule fragShaderModule = vks::tools::loadShader("shaders/frag.spirv", device->logicalDevice);

    VkPipelineShaderStageCreateInfo vertShaderStageCI{};
    vertShaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageCI.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageCI.module = vertShaderModule;
    vertShaderStageCI.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageCI{};
    fragShaderStageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageCI.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageCI.module = fragShaderModule;
    fragShaderStageCI.pName = "main";

    std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
        vertShaderStageCI,
        fragShaderStageCI
    };

    auto bindingDescription = Vertex::getBindingDescription();
    auto attribDescription = Vertex::getAttributeDescriptions();

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attribDescription.size());
    vertexInputInfo.pVertexAttributeDescriptions = attribDescription.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyCI{};
    inputAssemblyCI.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyCI.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssemblyCI.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportStateCI{};
    viewportStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportStateCI.viewportCount = 1;
    viewportStateCI.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizerStateCI{};
    rasterizerStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizerStateCI.depthClampEnable = VK_FALSE;
    rasterizerStateCI.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizerStateCI.lineWidth = 1.0f;
    //rasterizerStateCI.cullMode = VK_CULL_MODE_FRONT_BIT; // technically back bit since the y-axis is flipped when viewport is set for draw call
    rasterizerStateCI.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizerStateCI.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizerStateCI.depthBiasEnable = VK_FALSE;
    rasterizerStateCI.depthBiasConstantFactor = 0.0f;
    rasterizerStateCI.depthBiasClamp = 0.0f;
    rasterizerStateCI.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisamplingStateCI{};
    multisamplingStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisamplingStateCI.sampleShadingEnable = VK_FALSE;
    multisamplingStateCI.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisamplingStateCI.minSampleShading = 1.0f;
    multisamplingStateCI.pSampleMask = nullptr;
    multisamplingStateCI.alphaToCoverageEnable = VK_FALSE;
    multisamplingStateCI.alphaToOneEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

    VkPipelineColorBlendStateCreateInfo colorBlendingStateCI{};
    colorBlendingStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendingStateCI.logicOpEnable = VK_FALSE;
    colorBlendingStateCI.logicOp = VK_LOGIC_OP_COPY; // Optional;
    colorBlendingStateCI.attachmentCount = 1;
    colorBlendingStateCI.pAttachments = &colorBlendAttachment;
    colorBlendingStateCI.blendConstants[0] = 0.0f; // Optional
    colorBlendingStateCI.blendConstants[1] = 0.0f; // Optional
    colorBlendingStateCI.blendConstants[2] = 0.0f; // Optional
    colorBlendingStateCI.blendConstants[3] = 0.0f; // Optional

    VkPipelineDepthStencilStateCreateInfo depthStencilStateCI{};
    depthStencilStateCI.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilStateCI.depthTestEnable = VK_TRUE;
    depthStencilStateCI.depthWriteEnable = VK_TRUE;
    depthStencilStateCI.depthCompareOp = VK_COMPARE_OP_LESS;

    depthStencilStateCI.depthBoundsTestEnable = VK_FALSE;
    depthStencilStateCI.minDepthBounds = 0.0f; // Optional
    depthStencilStateCI.maxDepthBounds = 1.0f; // Optional

    depthStencilStateCI.stencilTestEnable = VK_FALSE;
    depthStencilStateCI.front = {}; // Optional
    depthStencilStateCI.back = {}; // Optional

    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkFormat depthStencilFormat{};
    vks::tools::getSupportedDepthStencilFormat(device->physicalDevice, &depthStencilFormat);

    // dynamic rendering info
    VkPipelineRenderingCreateInfoKHR pipelineRenderingCI{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR };
    pipelineRenderingCI.colorAttachmentCount = 1;
    pipelineRenderingCI.pColorAttachmentFormats = &swapchain->colorFormat;
    pipelineRenderingCI.depthAttachmentFormat = depthStencilFormat;
    pipelineRenderingCI.stencilAttachmentFormat = depthStencilFormat;

    VkGraphicsPipelineCreateInfo pipelineCI{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pipelineCI.layout = graphicsPipelineLayout;
    pipelineCI.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCI.pStages = shaderStages.data();
    pipelineCI.pVertexInputState = &vertexInputInfo;
    pipelineCI.pInputAssemblyState = &inputAssemblyCI;
    pipelineCI.pViewportState = &viewportStateCI;
    pipelineCI.pRasterizationState = &rasterizerStateCI;
    pipelineCI.pMultisampleState = &multisamplingStateCI;
    pipelineCI.pDepthStencilState = &depthStencilStateCI;
    pipelineCI.pColorBlendState = &colorBlendingStateCI;
    pipelineCI.pDynamicState = &dynamicState;
    pipelineCI.pNext = &pipelineRenderingCI;

    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device->logicalDevice, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &graphicsPipeline));

    vkDestroyShaderModule(device->logicalDevice, vertShaderModule, nullptr);
    vkDestroyShaderModule(device->logicalDevice, fragShaderModule, nullptr);

    // allocate descriptor sets
    graphicsDescriptors.resize(MAX_CONCURRENT_FRAMES);
    std::vector<VkDescriptorSetLayout> layouts(MAX_CONCURRENT_FRAMES, graphicsDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = MAX_CONCURRENT_FRAMES;
    allocInfo.pSetLayouts = layouts.data();

    VK_CHECK_RESULT(vkAllocateDescriptorSets(device->logicalDevice, &allocInfo, graphicsDescriptors.data()));

    updateGraphicsDescriptors();
}



void Engine::updateGraphicsDescriptors()
{
    for (uint32_t i = 0; i < MAX_CONCURRENT_FRAMES; i++)
    {
        VkDescriptorBufferInfo uboInfo{};
        uboInfo.buffer = graphicsUBO[i].buffer;
        uboInfo.offset = 0;
        uboInfo.range = sizeof(MVPMatrices);


        std::array<VkWriteDescriptorSet, 1> descriptorWrites{};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = graphicsDescriptors[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &uboInfo;



        vkUpdateDescriptorSets(device->logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
    }
}

void Engine::cleanUpGraphicsResources()
{
    if (graphicsPipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device->logicalDevice, graphicsPipeline, nullptr);
        graphicsPipeline = VK_NULL_HANDLE;
    }
    if (graphicsPipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device->logicalDevice, graphicsPipelineLayout, nullptr);
        graphicsPipelineLayout = VK_NULL_HANDLE;
    }
    if (graphicsDescriptorSetLayout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device->logicalDevice, graphicsDescriptorSetLayout, nullptr);
        graphicsDescriptorSetLayout = VK_NULL_HANDLE;
    }
    for (auto buffer : graphicsUBO) buffer.destroy();
}

void Engine::createDepthResources()
{
    VkFormat depthStencilFormat{};
    vks::tools::getSupportedDepthStencilFormat(device->physicalDevice, &depthStencilFormat);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = swapchain->extent.width;
    imageInfo.extent.height = swapchain->extent.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = depthStencilFormat;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.flags = 0;

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = depthStencilFormat;

    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    depthStencil.imageInfo = imageInfo;
    depthStencil.viewInfo = viewInfo;

    depthStencil.createImage(device->logicalDevice, device->physicalDevice, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void Engine::initializeVertexIndexBuffers()
{

    VkDeviceSize vertexBufferSize = sizeof(Vertex) * heightMapSize * heightMapSize;

    VkDeviceSize indexBufferSize = sizeof(uint32_t) * (heightMapSize - 1) * (heightMapSize - 1) * 6;

    vertexBuffer.create(
        device->logicalDevice,
        device->physicalDevice,
        vertexBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    indexBuffer.create(
        device->logicalDevice,
        device->physicalDevice,
        indexBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
}

void Engine::cleanUpVertexIndexBuffers()
{
    vertexBuffer.destroy();
    indexBuffer.destroy();
}

void Engine::createHeightMapResources(int size, VkFormat format)
{
    VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = size;
    imageInfo.extent.height = size;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    heightMap.imageInfo = imageInfo;
    heightMap.viewInfo = viewInfo;

    heightMap.createImage(device->logicalDevice, device->physicalDevice, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings(1);
    layoutBindings[0].binding = 0;
    layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    layoutBindings[0].descriptorCount = 1;
    layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    m_heightMapCompute = std::make_unique<VulkanComputePass>(*device);
    VulkanComputePass::Config computeConfig{};
    computeConfig.descriptorSetLayoutBindings = layoutBindings;
    computeConfig.shaderPath = "shaders/heightmap.spirv";
    computeConfig.shaderType = VulkanComputePass::ShaderType::Shader_Type_SPIRV;
    computeConfig.slangGlobalSession = nullptr;
    computeConfig.pushConstantSize = sizeof(HeightMapParams);
    m_heightMapCompute->create(computeConfig, descriptorPool);

    VkDescriptorImageInfo storageImageDescriptor{};
    storageImageDescriptor.imageView = heightMap.imageView;
    storageImageDescriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::vector<VkWriteDescriptorSet> writeDescriptorSets(1);
    writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSets[0].dstBinding = 0;
    //writeDescriptorSets[0].dstSet = heightMapDescriptors;
    writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeDescriptorSets[0].descriptorCount = 1;
    writeDescriptorSets[0].pImageInfo = &storageImageDescriptor;

    m_heightMapCompute->updateDescriptors(writeDescriptorSets);
}

void Engine::createTerrainGenerationComputeResources()
{
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);
    // height map
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // vertex buffer
    bindings[1].binding = 1;
    bindings[1].descriptorCount = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // index buffer
    bindings[2].binding = 2;
    bindings[2].descriptorCount = 1;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    m_TerrainGenerationCompute = std::make_unique<VulkanComputePass>(*device);
    VulkanComputePass::Config computeConfig{};
    computeConfig.descriptorSetLayoutBindings = bindings;
    computeConfig.shaderPath = "shaders/GenerateTerrainMesh.spirv";
    computeConfig.shaderType = VulkanComputePass::ShaderType::Shader_Type_SPIRV;
    computeConfig.slangGlobalSession = nullptr;
    computeConfig.pushConstantSize = sizeof(TerrainParams);
    m_TerrainGenerationCompute->create(computeConfig, descriptorPool);

    VkDescriptorImageInfo heightMapInfo{};
    heightMapInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    heightMapInfo.imageView = heightMap.imageView;
    heightMapInfo.sampler = heightMap.sampler;

    VkDeviceSize vertexBufferSize = sizeof(Vertex) * heightMapSize * heightMapSize;
    VkDeviceSize indexBufferSize = sizeof(uint32_t) * (heightMapSize - 1) * (heightMapSize - 1) * 6;

    VkDescriptorBufferInfo vertexBufferInfo{};
    vertexBufferInfo.buffer = vertexBuffer.buffer;
    vertexBufferInfo.range = vertexBufferSize;
    vertexBufferInfo.offset = 0;

    VkDescriptorBufferInfo indexBufferInfo{};
    indexBufferInfo.buffer = indexBuffer.buffer;
    indexBufferInfo.range = indexBufferSize;
    indexBufferInfo.offset = 0;

    std::vector<VkWriteDescriptorSet> writeDescriptorSets(3);
    writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSets[0].dstBinding = 0;
    //writeDescriptorSets[0].dstSet = terrainGenDescriptors;
    writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writeDescriptorSets[0].descriptorCount = 1;
    writeDescriptorSets[0].pImageInfo = &heightMapInfo;

    writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSets[1].dstBinding = 1;
    //writeDescriptorSets[1].dstSet = terrainGenDescriptors;
    writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSets[1].descriptorCount = 1;
    writeDescriptorSets[1].pBufferInfo = &vertexBufferInfo;

    writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSets[2].dstBinding = 2;
    //writeDescriptorSets[2].dstSet = terrainGenDescriptors;
    writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSets[2].descriptorCount = 1;
    writeDescriptorSets[2].pBufferInfo = &indexBufferInfo;


    m_TerrainGenerationCompute->updateDescriptors(writeDescriptorSets);

}

void Engine::cleanUpTerrainGenerationComputeResources()
{

    m_TerrainGenerationCompute.reset();
}

// combines heightmap generation and mesh generation
void Engine::recordTerrainMeshGeneration(VkCommandBuffer cmd, HeightMapParams& heightMapParams, TerrainParams& terrainParams)
{
    VkImageLayout oldLayout = heightMapInitialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    VkAccessFlags srcAccessMask = heightMapInitialized ? VK_ACCESS_SHADER_READ_BIT : 0;
    VkPipelineStageFlags srcStage = heightMapInitialized ? VK_PIPELINE_STAGE_VERTEX_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;


    // height map generation
    vks::tools::insertImageMemoryBarrier(
        cmd,
        heightMap.image,
        srcAccessMask,
        VK_ACCESS_SHADER_WRITE_BIT,
        oldLayout,
        VK_IMAGE_LAYOUT_GENERAL,
        srcStage,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    /*vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, heightMapComputePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, heightMapPipelineLayout, 0, 1, &heightMapDescriptors, 0, nullptr);
    vkCmdPushConstants(cmd, heightMapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HeightMapParams), &heightMapParams);*/

    uint32_t groupSize = 8;
    uint32_t gx = (heightMapSize + groupSize - 1) / groupSize;
    uint32_t gy = gx;
    //vkCmdDispatch(cmd, gx, gy, 1);

    m_heightMapCompute->recordCommands(cmd, &heightMapParams, gx, gy, 1);

    vks::tools::insertImageMemoryBarrier(
        cmd,
        heightMap.image,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    VkDeviceSize vertexBufferSize = sizeof(Vertex) * heightMapSize * heightMapSize;
    VkDeviceSize indexBufferSize = sizeof(uint32_t) * (heightMapSize - 1) * (heightMapSize - 1) * 6;

    VkAccessFlags vertexSrcAccess = heightMapInitialized ? VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT : 0;
    VkPipelineStageFlags vertexSrcStage = heightMapInitialized ? VK_PIPELINE_STAGE_VERTEX_INPUT_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkAccessFlags indexSrcAccess = heightMapInitialized ? VK_ACCESS_INDEX_READ_BIT : 0;
    VkPipelineStageFlags indexSrcStage = heightMapInitialized ? VK_PIPELINE_STAGE_VERTEX_INPUT_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    vks::tools::insertBufferMemoryBarrier(
        cmd,
        vertexSrcAccess,
        VK_ACCESS_SHADER_WRITE_BIT,
        vertexBuffer.buffer,
        0,
        vertexBufferSize,
        vertexSrcStage,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );
    vks::tools::insertBufferMemoryBarrier(
        cmd,
        indexSrcAccess,
        VK_ACCESS_SHADER_WRITE_BIT,
        indexBuffer.buffer,
        0,
        indexBufferSize,
        indexSrcStage,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );

    m_TerrainGenerationCompute->recordCommands(cmd, &terrainParams, gx, gy, 1);

    vks::tools::insertBufferMemoryBarrier(
        cmd,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
        vertexBuffer.buffer,
        0,
        vertexBufferSize,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
    );

    vks::tools::insertBufferMemoryBarrier(
        cmd,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_INDEX_READ_BIT,
        indexBuffer.buffer,
        0,
        indexBufferSize,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
    );

    generationCalls++;

    heightMapConfigChanged = false;
    heightMapInitialized = true;
}


void Engine::cleanUpHeightMapResources()
{
    heightMap.destroy();

    m_heightMapCompute.reset();
}


void Engine::debugPrintIndexBuffer()
{
    // Make sure all GPU commands are finished before we try to copy the buffer
    vkDeviceWaitIdle(device->logicalDevice);

    // Assuming your VulkanBuffer struct/class stores its size.
    // If not, calculate it manually:
    const VkDeviceSize bufferSize = sizeof(uint32_t) * (heightMapSize - 1) * (heightMapSize - 1) * 6;

    // 1. Create a staging buffer that is visible to the CPU (the host)
    vks::Buffer stagingBuffer;
    stagingBuffer.create(
        device->logicalDevice,
        device->physicalDevice,
        bufferSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, // We're using it as a destination for a copy
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    const VkDeviceSize vertSize = sizeof(Vertex) * heightMapSize * heightMapSize;
    vks::Buffer stagingBufferVert;
    stagingBufferVert.create(
        device->logicalDevice,
        device->physicalDevice,
        vertSize,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, // We're using it as a destination for a copy
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    // 2. Use a command buffer to record the copy command
    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = device->graphicsCommandPool; // Using your existing graphics command pool
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer copyCmd;
    vkAllocateCommandBuffers(device->logicalDevice, &allocInfo, &copyCmd);

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; // We'll only use this command buffer once
    vkBeginCommandBuffer(copyCmd, &beginInfo);

    // Record the actual copy command
    VkBufferCopy copyRegion{};
    copyRegion.size = bufferSize;
    vkCmdCopyBuffer(copyCmd, indexBuffer.buffer, stagingBuffer.buffer, 1, &copyRegion);

    copyRegion.size = vertSize;
    vkCmdCopyBuffer(copyCmd, vertexBuffer.buffer, stagingBufferVert.buffer, 1, &copyRegion);

    vkEndCommandBuffer(copyCmd);

    // 3. Submit the command to the queue and wait for it to finish
    VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &copyCmd;

    vkQueueSubmit(device->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(device->graphicsQueue); // The simplest way to wait for the copy to complete

    vkFreeCommandBuffers(device->logicalDevice, device->graphicsCommandPool, 1, &copyCmd);


    stagingBufferVert.map();
    if (stagingBufferVert.mapped)
    {
        // Cast the raw data pointer to a Vertex pointer
        Vertex* vertices = static_cast<Vertex*>(stagingBufferVert.mapped);
        uint32_t vertexCount = vertSize / sizeof(Vertex);

        std::cout << "\n--- Vertex Buffer Positions (" << vertexCount << " vertices) ---" << std::endl;
        // Loop through all the vertices
        for (uint32_t i = 0; i < vertexCount; ++i)
        {
            // Access the 'pos' member of the current vertex
            glm::vec3 position = vertices[i].pos;
            std::cout << "V" << i << ": ("
                << position.x << ", "
                << position.y << ", "
                << position.z << ")" << std::endl;
        }
        std::cout << "\n--------------------------------------------------\n" << std::endl;

        stagingBufferVert.unmap();
    }
    else
    {
        std::cerr << "Failed to map vertex staging buffer memory!" << std::endl;
    }


    // 4. Map the staging buffer's memory, read it, and print
    stagingBuffer.map();
    if (stagingBuffer.mapped)
    {
        uint32_t* indices = static_cast<uint32_t*>(stagingBuffer.mapped);
        uint32_t indexCount = bufferSize / sizeof(uint32_t);

        std::cout << "\n--- Index Buffer Contents (" << indexCount << " indices) ---" << std::endl;
        for (uint32_t i = 0; i < indexCount; ++i)
        {
            std::cout << indices[i] << " ";
            // Add a newline every 6 indices (for one quad) to make it readable
            if ((i + 1) % 6 == 0)
            {
                std::cout << std::endl;
            }
        }
        std::cout << "\n--------------------------------------------\n" << std::endl;

        stagingBuffer.unmap();
    }
    else
    {
        std::cerr << "Failed to map staging buffer memory!" << std::endl;
    }

    // 5. Clean up the temporary staging buffer
    stagingBuffer.destroy();
}