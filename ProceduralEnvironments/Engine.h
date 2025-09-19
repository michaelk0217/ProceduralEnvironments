#pragma once

#include <memory>
#include <chrono>
#include <vector>

#include <vulkan/vulkan.h>
#include <slang/slang.h>

#include "VulkanDevice.h"
#include "VulkanSwapchain.h"
#include "VulkanBuffer.h"
#include "VulkanImage.h"

#include "VulkanStructures.h"

#include "Window.h"
#include "Camera.hpp"
#include "UIOverlay.h"

class Engine
{
public:
	void run();

private:
	struct {
		uint32_t width = 1960;
		uint32_t height = 1080;
		const std::string windowTitle = "Engine";
		bool resized = false;
	} windowConfig;

	std::unique_ptr<UIOverlay> uiOverlay;

	uint32_t currentFrame = 0;
	static const uint32_t MAX_CONCURRENT_FRAMES = 2;
	float totalElapsedTime = 0.0f;
	std::vector<float> frame_history;

	slang::IGlobalSession* slangGlobalSession;

	std::unique_ptr<Window> window;
	std::unique_ptr<Camera> camera;
	std::unique_ptr<VulkanDevice> device;
	std::unique_ptr<VulkanSwapchain> swapchain;

	std::vector<VkCommandBuffer> frameCommandBuffers;

	void initGlfwWindow();
	void initVulkan();
	void mainLoop();
	void cleanUp();

	void drawFrame();
	void windowResize();
	void processInput(float deltaTime);

	// ----- Sync Objects -----
	void createSyncPrimitives();
	std::vector<VkSemaphore> presentCompleteSemaphores{};
	std::vector<VkSemaphore> renderCompleteSemaphores{};
	std::array<VkFence, MAX_CONCURRENT_FRAMES> waitFences{};
	void cleanUpSyncPrimitives();

	// ----- Descriptor Pool -----
	void createDescriptorPool();
	VkDescriptorPool descriptorPool;

	// ----- Graphics Pipeline -----
	void createGraphicsResources();
	VkDescriptorSetLayout graphicsDescriptorSetLayout;
	VkPipelineLayout graphicsPipelineLayout;
	VkPipeline graphicsPipeline;
	void updateGraphicsDescriptors();
	std::vector<VkDescriptorSet> graphicsDescriptors;
	std::vector<vks::Buffer> graphicsUBO;
	void cleanUpGraphicsResources();

	// ----- Depth/Stencil Resources -----
	void createDepthResources();
	vks::Image depthStencil;

	// ----- Vertex / Index Buffers -----
	void initializeVertexIndexBuffers();
	vks::Buffer vertexBuffer;
	vks::Buffer indexBuffer;
	std::vector<Vertex> triangleVertices = {
		{
			glm::vec3(1.0f, -1.0f, 0.0f),    // pos
			glm::vec3(0.0f, 0.0f, 1.0f),     // inNormal (pointing towards camera)
			glm::vec2(0.5f, 0.0f),           // texCoord
			glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)      // color (red)
		},
		{
			glm::vec3(-1.0f, -1.0f, 0.0f),    // pos
			glm::vec3(0.0f, 0.0f, 1.0f),     // inNormal
			glm::vec2(0.0f, 1.0f),           // texCoord
			glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)      // color (blue)
		},
		{
			glm::vec3(0.0f, 1.0f, 0.0f),     // pos
			glm::vec3(0.0f, 0.0f, 1.0f),     // inNormal
			glm::vec2(1.0f, 1.0f),           // texCoord
			glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)      // color (green)
		}
	};
	std::vector<uint32_t> triangleIndices{ 0, 1, 2 };

	std::vector<Vertex> quadsVertices{};
	std::vector<uint32_t> quadsIndices{};
	void cleanUpVertexIndexBuffers();


	// ----- utility funcitons -----
	static void generateNxNQuad(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, uint32_t n, float l);

	void createHeightMapResources(int size, VkFormat format);
	VkDescriptorSetLayout heightMapDescriptorSetLayout;
	VkPipelineLayout heightMapPipelineLayout;
	VkPipeline heightMapComputePipeline;
	VkDescriptorSet heightMapDescriptors;
	vks::Image heightMap;
	VkSampler heightMapSampler;
	vks::Buffer heightMapParams;

	int heightMapSize = 512;
	HeightMapParams heightMapConfig{
		12345, // seed
		glm::vec2(0.0), // offset
		1.0f, // frequency
		6, // octaves
		2.0, // lacunarity
		0.5 // persistnece
	};
	bool heightMapConfigChanged = true;
	bool heightMapInitialized = false;

	void generateHeightMap(HeightMapParams& params, int size);
	void cleanUpHeightMapResources();

};

