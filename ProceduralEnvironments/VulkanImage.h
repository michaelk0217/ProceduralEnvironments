#pragma once

#include <vulkan/vulkan.h>

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



		void createImage(VkDevice device, VkPhysicalDevice physicalDevice, VkMemoryPropertyFlags properties);
		void destroy();
	};
}