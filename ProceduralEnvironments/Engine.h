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
#include "VulkanComputePass.h"
#include "VulkanStructures.h"

#include "Window.h"
#include "Camera.hpp"
#include "UIOverlay.h"

//#include "GpuCrashTracker.h"

class Engine
{
public:
	void run();

private:
	// ---- debug helperts -----
	uint32_t generationCalls = 0;
	void debugPrintIndexBuffer();

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
	VertexShaderPushConstant vertPushConstant;
	void cleanUpGraphicsResources();

	// ----- Depth/Stencil Resources -----
	void createDepthResources();
	vks::Image depthStencil;

	// ----- Vertex / Index Buffers -----
	void initializeVertexIndexBuffers();
	vks::Buffer vertexBuffer;
	vks::Buffer indexBuffer;

	std::vector<Vertex> quadsVertices{};
	std::vector<uint32_t> quadsIndices{};
	void cleanUpVertexIndexBuffers();


	// ----- Height Map -----
	void createHeightMapResources(int size, VkFormat format);
	vks::Image heightMap;
	int heightMapSize = 1024;
	HeightMapParams heightMapConfig;
	
	std::unique_ptr<VulkanComputePass> m_heightMapCompute;

	void cleanUpHeightMapResources();


	
	// generate heightmap AND normals
	bool heightMapConfigChanged = true;
	bool heightMapInitialized = false;

	// generate terrain mesh
	void createTerrainGenerationComputeResources();
	TerrainParams terrainGenParams;
	std::unique_ptr<VulkanComputePass> m_TerrainGenerationCompute;
	void cleanUpTerrainGenerationComputeResources();
	void recordTerrainMeshGeneration(VkCommandBuffer cmd, HeightMapParams& heightMapParams, TerrainParams& terrainParams);
};

