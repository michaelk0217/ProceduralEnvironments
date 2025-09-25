#pragma once

#include "VulkanDevice.h"
#include <slang/slang.h>
#include <string>
#include <vector>


class VulkanComputePass
{
public:
	
	enum class ShaderType {
		Shader_Type_SPIRV,
		Shader_Type_SLANG
	};

	struct Config {
		std::string shaderPath;
		ShaderType shaderType;
		slang::IGlobalSession* slangGlobalSession;
		std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings;
		uint32_t pushConstantSize = 0;
	};

	VulkanComputePass(VulkanDevice& device);
	~VulkanComputePass();

	VulkanComputePass(const VulkanComputePass&) = delete;
	VulkanComputePass& operator=(const VulkanComputePass&) = delete;

	/** @brief Create Compute Pass with all vulkan resources based on the config */
	void create(const Config& config, VkDescriptorPool descriptorPool);

	/* 
	* @brief Update the descriptor set with specific buffers / images. This must be called after create()
	*/
	void updateDescriptors(const std::vector<VkWriteDescriptorSet>& descriptorWrites);

	/* @brief Record the compute dispatch commands into a command buffer */
	void recordCommands(
		VkCommandBuffer cmd,
		const void* pushConstantData,
		uint32_t dispatchGroupX,
		uint32_t dispatchGroupY,
		uint32_t dispatchGroupZ
	);

	/* @brief Get descriptor set */
	VkDescriptorSet getDescriptorSet() const { return descriptorSet; }

private:
	const VulkanDevice& device;
	Config config;
	
	VkPipeline computePipeline = VK_NULL_HANDLE;
	VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
	VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
	VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
};

