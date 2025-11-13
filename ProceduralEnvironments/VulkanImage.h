#pragma once

#include <vulkan/vulkan.h>

#include "VulkanDevice.h"
#include <string>
#include <stb_image.h>


namespace vks
{
	struct Image
	{
		VkDevice device;
		VkImage image = VK_NULL_HANDLE;
		VkImageView imageView = VK_NULL_HANDLE;
		VkDeviceMemory memory = VK_NULL_HANDLE;
		VkSampler sampler = VK_NULL_HANDLE;

		VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };

		VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };

		void createImage(VkDevice device, VkPhysicalDevice physicalDevice, VkMemoryPropertyFlags properties, bool createSampler = true);

		void loadFromFile(VulkanDevice& device, std::string path, VkFormat format, bool createSampler = true);

		void destroy();

		static void transferHdrDataToImage(VulkanDevice& device, float* pixelData, VkImage image, uint32_t width, uint32_t height, VkPipelineStageFlagBits imageDstStage);

		static uint32_t getBytesPerPixel(VkFormat format);
	};
}