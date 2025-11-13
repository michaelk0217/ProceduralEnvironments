#include "PBRTexture.h"

PBRTexture::PBRTexture() : 
	sampler(VK_NULL_HANDLE)
{
}

PBRTexture::~PBRTexture()
{
	color.destroy();
	ambientOcclusion.destroy();
	normal.destroy();
	roughness.destroy();
	displacement.destroy();
}

void PBRTexture::initialize(VulkanDevice& device, std::string colorPath, std::string aoPath, std::string normalPath, std::string roughnessPath, std::string displacementPath)
{
	color.loadFromFile(device, colorPath, VK_FORMAT_R8G8B8A8_SRGB, false);
	ambientOcclusion.loadFromFile(device, aoPath, VK_FORMAT_R8G8B8A8_SRGB, false);
	normal.loadFromFile(device, normalPath, VK_FORMAT_R8G8B8A8_SRGB, false);
	roughness.loadFromFile(device, roughnessPath, VK_FORMAT_R8G8B8A8_SRGB, false);
	displacement.loadFromFile(device, displacementPath, VK_FORMAT_R8G8B8A8_SRGB, false);

	VkSamplerCreateInfo samplerCI{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
	samplerCI.magFilter = VK_FILTER_LINEAR;
	samplerCI.minFilter = VK_FILTER_LINEAR;
	samplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCI.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerCI.maxAnisotropy = 16.0f;
	samplerCI.anisotropyEnable = VK_TRUE;
	samplerCI.maxLod = FLT_MAX;
	VK_CHECK_RESULT(vkCreateSampler(device.logicalDevice, &samplerCI, nullptr, &sampler));
}
	