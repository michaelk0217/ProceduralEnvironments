#pragma once

#include <vulkan/vulkan.h>

#include "VulkanImage.h"
#include "VulkanDevice.h"
#include "VulkanTools.h"
#include <string>

class PBRTexture
{
public:
	PBRTexture();
	~PBRTexture();

	void initialize(VulkanDevice& device, std::string colorPath, std::string aoPath, std::string normalPath, std::string roughnessPath, std::string displacementPath);



private:
	vks::Image color;
	vks::Image ambientOcclusion;
	vks::Image normal;
	vks::Image roughness;
	vks::Image displacement;

	VkSampler sampler;
};

