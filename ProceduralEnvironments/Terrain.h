#pragma once

#include <memory>
#include <vulkan/vulkan.h>
#include "VulkanDevice.h"
#include "VulkanBuffer.h"
#include "VulkanImage.h"
#include "VulkanComputePass.h"
#include "VulkanStructures.h"

class Terrain
{
public:
	struct Config {
        uint32_t heightmapSize = 1024;
        uint32_t gridResolution = 1024;
        float terrainSideLength = 40.0f;
        float heightScale = 3.0f;
        float normalsStrength = 50.0f;
        VkFormat heightmapFormat = VK_FORMAT_R32_SFLOAT;
	};

    Terrain(VulkanDevice& device, const Config& config);
    ~Terrain();

    Terrain(const Terrain&) = delete;
    Terrain& operator=(const Terrain&) = delete;

    /**
     * @brief Initialize all Vulkan resources (buffers, images, compute pipelines)
     * @param descriptorPool Pool to allocate descriptor sets from
     */
    void initialize(VkDescriptorPool descriptorPool);

    /**
     * @brief Record terrain generation commands into command buffer
     * @param cmd Command buffer to record into
     * @param heightMapParams Parameters for heightmap generation
     * @param terrainParams Parameters for mesh generation
     */
    void recordGeneration(VkCommandBuffer cmd, const HeightMapParams& heightMapParams, const TerrainParams& terrainParams);

    /**
     * @brief Record draw commands for the terrain
     * @param cmd Command buffer to record into
     */
    void recordDraw(VkCommandBuffer cmd);

    /**
     * @brief Get whether the terrain has been generated at least once
     */
    bool isInitialized() const { return m_initialized; }

    /**
     * @brief Mark terrain as needing regeneration
     */
    void markDirty() { m_initialized = false; }

    // Getters
    const vks::Buffer& getVertexBuffer() const { return m_vertexBuffer; }
    const vks::Buffer& getIndexBuffer() const { return m_indexBuffer; }
    const vks::Image& getHeightmap() const { return m_heightMap; }
    uint32_t getIndexCount() const { return m_indexCount; }
    const Config& getConfig() const { return m_config; }

    // Debug utilities
    void debugPrintBuffers() const;

private:
    void createHeightmapResources();
    void createMeshBuffers();
    void createHeightmapComputePass(VkDescriptorPool descriptorPool);
    void createTerrainGenComputePass(VkDescriptorPool descriptorPool);

    void cleanup();

    VulkanDevice& m_device;
    Config m_config;

    // Heightmap resources
    vks::Image m_heightMap;
    std::unique_ptr<VulkanComputePass> m_heightMapCompute;

    // Mesh resources
    vks::Buffer m_vertexBuffer;
    vks::Buffer m_indexBuffer;
    uint32_t m_indexCount;
    std::unique_ptr<VulkanComputePass> m_terrainGenCompute;

    // State
    bool m_initialized = false;
    uint32_t m_generationCount = 0;
};

