#include "VulkanComputePass.h"

#include "VulkanTools.h"

VulkanComputePass::VulkanComputePass(VulkanDevice& device)
	: device(device)
{
}

VulkanComputePass::~VulkanComputePass()
{
    if (computePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device.logicalDevice, computePipeline, nullptr);
    }
    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device.logicalDevice, pipelineLayout, nullptr);
    }
    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device.logicalDevice, descriptorSetLayout, nullptr);
    }
}

void VulkanComputePass::create(const Config& config, VkDescriptorPool descriptorPool)
{
    this->config = config;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCI{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    descriptorSetLayoutCI.bindingCount = static_cast<uint32_t>(this->config.descriptorSetLayoutBindings.size());
    descriptorSetLayoutCI.pBindings = this->config.descriptorSetLayoutBindings.data();
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(this->device.logicalDevice, &descriptorSetLayoutCI, nullptr, &this->descriptorSetLayout));

    VkPushConstantRange pushConstant{};
    pushConstant.offset = 0;
    pushConstant.size = this->config.pushConstantSize;
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkPipelineLayoutCreateInfo pipelineLayoutCI{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutCI.setLayoutCount = 1;
    pipelineLayoutCI.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutCI.pushConstantRangeCount = this->config.pushConstantSize == 0 ? 0 : 1;
    pipelineLayoutCI.pPushConstantRanges = this->config.pushConstantSize == 0 ? nullptr : &pushConstant;
    VK_CHECK_RESULT(vkCreatePipelineLayout(this->device.logicalDevice, &pipelineLayoutCI, nullptr, &this->pipelineLayout));

    VkShaderModule computeShader;
    if (this->config.shaderType == ShaderType::Shader_Type_SPIRV)
    {
        computeShader = vks::tools::loadShader(this->config.shaderPath.c_str(), this->device.logicalDevice);
    }
    else if (this->config.shaderType == ShaderType::Shader_Type_SLANG)
    {
        computeShader = vks::tools::loadSlangShader(this->device.logicalDevice, this->config.slangGlobalSession, this->config.shaderPath.c_str(), "main");
    }
    else
    {
        throw std::runtime_error("Invalid Shader Type");
    }

    VkPipelineShaderStageCreateInfo shaderStage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    shaderStage.module = computeShader;
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.pName = "main";

    VkComputePipelineCreateInfo computeCI{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    computeCI.stage = shaderStage;
    computeCI.layout = this->pipelineLayout;
    VK_CHECK_RESULT(vkCreateComputePipelines(this->device.logicalDevice, VK_NULL_HANDLE, 1, &computeCI, nullptr, &this->computePipeline));

    vkDestroyShaderModule(this->device.logicalDevice, computeShader, nullptr);

    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &this->descriptorSetLayout;
    VK_CHECK_RESULT(vkAllocateDescriptorSets(this->device.logicalDevice, &allocInfo, &this->descriptorSet));
}

void VulkanComputePass::updateDescriptors(const std::vector<VkWriteDescriptorSet>& descriptorWrites)
{
    for (auto& write : const_cast<std::vector<VkWriteDescriptorSet>&>(descriptorWrites))
    {
        write.dstSet = this->descriptorSet;
    }
    vkUpdateDescriptorSets(this->device.logicalDevice, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
}

void VulkanComputePass::recordCommands(VkCommandBuffer cmd, const void* pushConstantData, uint32_t dispatchGroupX, uint32_t dispatchGroupY, uint32_t dispatchGroupZ)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->computePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, this->pipelineLayout, 0, 1, &this->descriptorSet, 0, nullptr);
    if (this->config.pushConstantSize > 0 && pushConstantData != nullptr)
    {
        vkCmdPushConstants(cmd, this->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, this->config.pushConstantSize, pushConstantData);
    }
    vkCmdDispatch(cmd, dispatchGroupX, dispatchGroupY, dispatchGroupZ);
}
