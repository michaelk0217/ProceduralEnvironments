#include "VulkanImage.h"
#include "VulkanTools.h"

#include "VulkanBuffer.h"

/**
* Creates Image, Imageview, and memory.
* Must have a complete imageInfo and viewInfo (createInfo) before calling this function.
*/
void vks::Image::createImage(VkDevice device, VkPhysicalDevice physicalDevice, VkMemoryPropertyFlags properties)
{
	this->device = device;
	VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &image));

	VkMemoryRequirements memRequirements;
	vkGetImageMemoryRequirements(device, image, &memRequirements);

	VkMemoryAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = memRequirements.size;
	allocInfo.memoryTypeIndex = vks::tools::findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

	VK_CHECK_RESULT(vkAllocateMemory(device, &allocInfo, nullptr, &memory));

	vkBindImageMemory(device, image, memory, 0);

	viewInfo.image = image;
	VK_CHECK_RESULT(vkCreateImageView(device, &viewInfo, nullptr, &imageView));

	VkSamplerCreateInfo samplerCI{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
	samplerCI.magFilter = VK_FILTER_LINEAR;
	samplerCI.minFilter = VK_FILTER_LINEAR;
	samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCI.anisotropyEnable = VK_FALSE;
	samplerCI.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
	samplerCI.unnormalizedCoordinates = VK_FALSE;
	samplerCI.compareEnable = VK_FALSE;
	samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerCI.minLod = 0.0f;
	samplerCI.maxLod = 0.0f;

	VK_CHECK_RESULT(vkCreateSampler(device, &samplerCI, nullptr, &sampler));
}

void vks::Image::destroy()
{
	if (image != VK_NULL_HANDLE)
	{
		vkDestroyImage(device, image, nullptr);
		image = VK_NULL_HANDLE;
	}
	if (imageView != VK_NULL_HANDLE)
	{
		vkDestroyImageView(device, imageView, nullptr);
		imageView = VK_NULL_HANDLE;
	}
	if (memory != VK_NULL_HANDLE)
	{
		vkFreeMemory(device, memory, nullptr);
		memory = VK_NULL_HANDLE;
	}
	if (sampler != VK_NULL_HANDLE)
	{
		vkDestroySampler(device, sampler, nullptr);
		sampler = VK_NULL_HANDLE;
	}
}

void vks::Image::transferHdrDataToImage(VulkanDevice& device, float* pixelData, VkImage image, uint32_t width, uint32_t height, VkPipelineStageFlagBits imageDstStage)
{
	// VK_FORMAT_R32G32B32A32_SFLOAT 
	VkDeviceSize imageSize = width * height * 4 * sizeof(float);

	vks::Buffer stagingBuffer;
	stagingBuffer.create(
		device.logicalDevice,
		device.physicalDevice,
		imageSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
	);
	stagingBuffer.map();
	stagingBuffer.copyTo(pixelData, imageSize);
	stagingBuffer.unmap();
	
	VkCommandBuffer cmd = vks::tools::beginSingleTimeCommands(device.logicalDevice, device.graphicsCommandPool);
	
	vks::tools::insertImageMemoryBarrier(
		cmd,
		image,
		0,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_IMAGE_LAYOUT_UNDEFINED,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
	);

	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;
	region.imageOffset = { 0, 0, 0 };
	region.imageExtent = { width, height, 1 };

	vkCmdCopyBufferToImage(
		cmd,
		stagingBuffer.buffer,
		image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1,
		&region
	);

	vks::tools::insertImageMemoryBarrier(
		cmd,
		image,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_SHADER_READ_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		imageDstStage,
		VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
	);

	vks::tools::endSingleTimeCommands(cmd, device.logicalDevice, device.graphicsQueue, device.graphicsCommandPool);

	stagingBuffer.destroy();
}
