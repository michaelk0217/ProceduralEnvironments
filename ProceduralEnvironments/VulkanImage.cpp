#include "VulkanImage.h"
#include "VulkanTools.h"

#include "VulkanBuffer.h"

/**
* Creates Image, Imageview, and memory.
* Must have a complete imageInfo and viewInfo (createInfo) before calling this function.
*/
void vks::Image::createImage(VkDevice device, VkPhysicalDevice physicalDevice, VkMemoryPropertyFlags properties, bool createSampler)
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

	if (createSampler)
	{
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
}

void vks::Image::loadFromFile(VulkanDevice& device, std::string path, VkFormat format, bool createSampler)
{
	int width, height, channels;
	
	unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);

	if (!data)
	{
		throw std::runtime_error("Failed to load image: " + path);
	}

	this->device = device.logicalDevice;

	imageInfo.format = format;
	imageInfo.imageType = VK_IMAGE_TYPE_2D;
	imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageInfo.extent.width = static_cast<uint32_t>(width);
	imageInfo.extent.height = static_cast<uint32_t>(height);
	imageInfo.extent.depth = 1;
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

	this->createImage(device.logicalDevice, device.physicalDevice, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, createSampler);

	uint32_t bytesPerPixel = getBytesPerPixel(format);
	VkDeviceSize imageSize = static_cast<uint32_t>(width) * static_cast<uint32_t>(height) * bytesPerPixel;

	vks::Buffer stagingBuffer;
	stagingBuffer.create(
		device.logicalDevice,
		device.physicalDevice,
		imageSize,
		VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
	);
	stagingBuffer.map();
	stagingBuffer.copyTo(data, imageSize);
	stagingBuffer.unmap();

	stbi_image_free(data);

	VkCommandBuffer cmd = vks::tools::beginSingleTimeCommands(device.logicalDevice, device.graphicsCommandPool);

	vks::tools::insertImageMemoryBarrier(
		cmd,
		this->image,
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
	region.imageExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1 };

	vkCmdCopyBufferToImage(
		cmd,
		stagingBuffer.buffer,
		this->image,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1,
		&region
	);

	vks::tools::insertImageMemoryBarrier(
		cmd,
		this->image,
		VK_ACCESS_TRANSFER_WRITE_BIT,
		VK_ACCESS_SHADER_READ_BIT,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		VK_PIPELINE_STAGE_TRANSFER_BIT,
		VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, // Or whatever stage will read it
		VkImageSubresourceRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
	);

	vks::tools::endSingleTimeCommands(cmd, device.logicalDevice, device.graphicsQueue, device.graphicsCommandPool);
	stagingBuffer.destroy();
}

uint32_t vks::Image::getBytesPerPixel(VkFormat format)
{
	switch (format)
	{
		// 8-bit formats (1 byte per pixel)
	case VK_FORMAT_R8_UNORM:
	case VK_FORMAT_R8_SNORM:
	case VK_FORMAT_R8_SRGB:
		return 1;

		// 16-bit formats (2 bytes per pixel)
	case VK_FORMAT_R16_SFLOAT:
	case VK_FORMAT_R8G8_UNORM:
	case VK_FORMAT_R8G8_SRGB:
		return 2;

		// 32-bit formats (4 bytes per pixel)
	case VK_FORMAT_R32_SFLOAT:
	case VK_FORMAT_R16G16_SFLOAT:
	case VK_FORMAT_R8G8B8A8_UNORM:
	case VK_FORMAT_R8G8B8A8_SNORM:
	case VK_FORMAT_R8G8B8A8_SRGB:
		return 4;

		// 64-bit formats (8 bytes per pixel)
	case VK_FORMAT_R32G32_SFLOAT:
	case VK_FORMAT_R16G16B16A16_SFLOAT:
		return 8;

		// 128-bit formats (16 bytes per pixel)
	case VK_FORMAT_R32G32B32A32_SFLOAT:
		return 16;

		// more formats maybe later

	default:
		// This doesn't handle compressed or depth/stencil formats
		throw std::runtime_error("Unsupported format for size calculation!");
	}
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
