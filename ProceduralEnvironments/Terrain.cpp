#include "Terrain.h"

Terrain::Terrain(VulkanDevice& device, const Config& config)
	: m_device(device)
	, m_config(config)
	, m_indexCount((config.gridResolution - 1)* (config.gridResolution - 1) * 6)
{
}

Terrain::~Terrain()
{
	cleanup();
}

void Terrain::initialize(VkDescriptorPool descriptorPool)
{
	createHeightmapResources();
	createMeshBuffers();
	createHeightmapComputePass(descriptorPool);
	createTerrainGenComputePass(descriptorPool);
}

void Terrain::recordGeneration(VkCommandBuffer cmd, const HeightMapParams& heightMapParams, const TerrainParams& terrainParams)
{
    VkImageLayout oldLayout = m_initialized ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    VkAccessFlags srcAccessMask = m_initialized ? VK_ACCESS_SHADER_READ_BIT : 0;
    VkPipelineStageFlags srcStage = m_initialized ? VK_PIPELINE_STAGE_VERTEX_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    // ============================================
    // HEIGHTMAP GENERATION
    // ============================================

    vks::tools::insertImageMemoryBarrier(
        cmd,
        m_heightMap.image,
        srcAccessMask,
        VK_ACCESS_SHADER_WRITE_BIT,
        oldLayout,
        VK_IMAGE_LAYOUT_GENERAL,
        srcStage,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    uint32_t groupSize = 8;
    uint32_t gx = (m_config.heightmapSize + groupSize - 1) / groupSize;
    uint32_t gy = gx;

    m_heightMapCompute->recordCommands(cmd, &heightMapParams, gx, gy, 1);

    vks::tools::insertImageMemoryBarrier(
        cmd,
        m_heightMap.image,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    );

    // ============================================
    // MESH GENERATION
    // ============================================

    VkDeviceSize vertexBufferSize = sizeof(Vertex) * m_config.gridResolution * m_config.gridResolution;
    VkDeviceSize indexBufferSize = sizeof(uint32_t) * m_indexCount;

    VkAccessFlags vertexSrcAccess = m_initialized ? VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT : 0;
    VkPipelineStageFlags vertexSrcStage = m_initialized ? VK_PIPELINE_STAGE_VERTEX_INPUT_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    VkAccessFlags indexSrcAccess = m_initialized ? VK_ACCESS_INDEX_READ_BIT : 0;
    VkPipelineStageFlags indexSrcStage = m_initialized ? VK_PIPELINE_STAGE_VERTEX_INPUT_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;

    vks::tools::insertBufferMemoryBarrier(
        cmd,
        vertexSrcAccess,
        VK_ACCESS_SHADER_WRITE_BIT,
        m_vertexBuffer.buffer,
        0,
        vertexBufferSize,
        vertexSrcStage,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );

    vks::tools::insertBufferMemoryBarrier(
        cmd,
        indexSrcAccess,
        VK_ACCESS_SHADER_WRITE_BIT,
        m_indexBuffer.buffer,
        0,
        indexBufferSize,
        indexSrcStage,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
    );

    m_terrainGenCompute->recordCommands(cmd, &terrainParams, gx, gy, 1);

    vks::tools::insertBufferMemoryBarrier(
        cmd,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
        m_vertexBuffer.buffer,
        0,
        vertexBufferSize,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
    );

    vks::tools::insertBufferMemoryBarrier(
        cmd,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_INDEX_READ_BIT,
        m_indexBuffer.buffer,
        0,
        indexBufferSize,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
    );

    m_generationCount++;
    m_initialized = true;
}

void Terrain::recordDraw(VkCommandBuffer cmd)
{
    VkDeviceSize offsets[1]{ 0 };
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer.buffer, offsets);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmd, m_indexCount, 1, 0, 0, 0);
}

void Terrain::createHeightmapResources()
{
    VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = m_config.heightmapSize;
    imageInfo.extent.height = m_config.heightmapSize;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = m_config.heightmapFormat;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = m_config.heightmapFormat;
    viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    m_heightMap.imageInfo = imageInfo;
    m_heightMap.viewInfo = viewInfo;

    m_heightMap.createImage(m_device.logicalDevice, m_device.physicalDevice, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

void Terrain::createMeshBuffers()
{
    VkDeviceSize vertexBufferSize = sizeof(Vertex) * m_config.gridResolution * m_config.gridResolution;
    VkDeviceSize indexBufferSize = sizeof(uint32_t) * m_indexCount;

    // Create vertex buffer
    m_vertexBuffer.create(
        m_device.logicalDevice,
        m_device.physicalDevice,
        vertexBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );

    // Create index buffer
    m_indexBuffer.create(
        m_device.logicalDevice,
        m_device.physicalDevice,
        indexBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    );
}

void Terrain::createHeightmapComputePass(VkDescriptorPool descriptorPool)
{
    std::vector<VkDescriptorSetLayoutBinding> layoutBindings(1);
    layoutBindings[0].binding = 0;
    layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    layoutBindings[0].descriptorCount = 1;
    layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    m_heightMapCompute = std::make_unique<VulkanComputePass>(m_device);
    VulkanComputePass::Config computeConfig{};
    computeConfig.descriptorSetLayoutBindings = layoutBindings;
    computeConfig.shaderPath = "shaders/heightmap.spirv";
    computeConfig.shaderType = VulkanComputePass::ShaderType::Shader_Type_SPIRV;
    computeConfig.slangGlobalSession = nullptr;
    computeConfig.pushConstantSize = sizeof(HeightMapParams);
    m_heightMapCompute->create(computeConfig, descriptorPool);

    // Update descriptors
    VkDescriptorImageInfo storageImageDescriptor{};
    storageImageDescriptor.imageView = m_heightMap.imageView;
    storageImageDescriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::vector<VkWriteDescriptorSet> writeDescriptorSets(1);
    writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSets[0].dstBinding = 0;
    writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeDescriptorSets[0].descriptorCount = 1;
    writeDescriptorSets[0].pImageInfo = &storageImageDescriptor;

    m_heightMapCompute->updateDescriptors(writeDescriptorSets);
}

void Terrain::createTerrainGenComputePass(VkDescriptorPool descriptorPool)
{
    // Descriptor layout: heightmap sampler, vertex buffer, index buffer
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);

    // Heightmap sampler
    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Vertex buffer
    bindings[1].binding = 1;
    bindings[1].descriptorCount = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    // Index buffer
    bindings[2].binding = 2;
    bindings[2].descriptorCount = 1;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    m_terrainGenCompute = std::make_unique<VulkanComputePass>(m_device);
    VulkanComputePass::Config computeConfig{};
    computeConfig.descriptorSetLayoutBindings = bindings;
    computeConfig.shaderPath = "shaders/GenerateTerrainMesh.spirv";
    computeConfig.shaderType = VulkanComputePass::ShaderType::Shader_Type_SPIRV;
    computeConfig.slangGlobalSession = nullptr;
    computeConfig.pushConstantSize = sizeof(TerrainParams);
    m_terrainGenCompute->create(computeConfig, descriptorPool);

    // Update descriptors
    VkDescriptorImageInfo heightMapInfo{};
    heightMapInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    heightMapInfo.imageView = m_heightMap.imageView;
    heightMapInfo.sampler = m_heightMap.sampler;

    VkDeviceSize vertexBufferSize = sizeof(Vertex) * m_config.gridResolution * m_config.gridResolution;
    VkDeviceSize indexBufferSize = sizeof(uint32_t) * m_indexCount;

    VkDescriptorBufferInfo vertexBufferInfo{};
    vertexBufferInfo.buffer = m_vertexBuffer.buffer;
    vertexBufferInfo.range = vertexBufferSize;
    vertexBufferInfo.offset = 0;

    VkDescriptorBufferInfo indexBufferInfo{};
    indexBufferInfo.buffer = m_indexBuffer.buffer;
    indexBufferInfo.range = indexBufferSize;
    indexBufferInfo.offset = 0;

    std::vector<VkWriteDescriptorSet> writeDescriptorSets(3);

    writeDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSets[0].dstBinding = 0;
    writeDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writeDescriptorSets[0].descriptorCount = 1;
    writeDescriptorSets[0].pImageInfo = &heightMapInfo;

    writeDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSets[1].dstBinding = 1;
    writeDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSets[1].descriptorCount = 1;
    writeDescriptorSets[1].pBufferInfo = &vertexBufferInfo;

    writeDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSets[2].dstBinding = 2;
    writeDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeDescriptorSets[2].descriptorCount = 1;
    writeDescriptorSets[2].pBufferInfo = &indexBufferInfo;

    m_terrainGenCompute->updateDescriptors(writeDescriptorSets);
}


void Terrain::debugPrintBuffers() const
{
    std::cout << "\n=== Terrain Debug Info ===" << std::endl;
    std::cout << "Heightmap Size: " << m_config.heightmapSize << "x" << m_config.heightmapSize << std::endl;
    std::cout << "Grid Resolution: " << m_config.gridResolution << std::endl;
    std::cout << "Index Count: " << m_indexCount << std::endl;
    std::cout << "Generation Count: " << m_generationCount << std::endl;
    std::cout << "Initialized: " << (m_initialized ? "Yes" : "No") << std::endl;
    std::cout << "=========================\n" << std::endl;
}

void Terrain::cleanup()
{
    m_terrainGenCompute.reset();
    m_heightMapCompute.reset();

    m_indexBuffer.destroy();
    m_vertexBuffer.destroy();
    m_heightMap.destroy();
}