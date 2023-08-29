﻿#define GLFW_INCLUDE_VULKAN
#include<GLFW/glfw3.h>
#include<iostream>
#include<stdexcept>
#include<cstdlib>
#include<cstring> // to use strcmp
#include<vector> // extesion prosperity
#include<optional>
#include<string>
#include<set>

#include <fstream>

#include <cstdint> // Necessary for uint32_t
#include <limits> // Necessary for std::numeric_limits
#include <algorithm> // Necessary for std::clamp

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono> // 时间
#include <array>

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif // NDEBUG

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 800;
const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };
const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
const int MAX_FRAMES_IN_FLIGHT = 2;
uint32_t currentFrame = 0;
const float PI = 3.14159;
const int OBJECT_INSTANCES = 10;

// def circle 
struct CirclePoint {
	glm::vec2 point;
	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bingdingDescription{};
		bingdingDescription.binding = 0;
		bingdingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		bingdingDescription.stride = sizeof(glm::vec2);
		return bingdingDescription;
	}
	static std::array< VkVertexInputAttributeDescription, 1> getAttributeDescriptions() {
		std::array< VkVertexInputAttributeDescription, 1> attributeDescriptions{};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[0].location = 0; // 和 shader 中的location相对应
		attributeDescriptions[0].offset = offsetof(CirclePoint, point); // 偏移量
		return attributeDescriptions;
	}
};

void generateCircle(float radius, int pointnum, std::vector<glm::vec2>& pointSet) {
	pointSet.resize(pointnum + 2);
	pointSet[0] = glm::vec2(0.0f, 0.0f);
	for (uint32_t i = 0; i <= pointnum; ++i) {
		pointSet[i + 1] = glm::vec2(radius * cos(2 * PI * i / pointnum), radius * sin(2 * PI * i / pointnum));
	}
}

std::vector<glm::vec2> circlePointSet;
// circle def end

struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;
	static VkVertexInputBindingDescription getBingdingDescrition() {
		VkVertexInputBindingDescription bingdingDescription{};
		bingdingDescription.binding = 0;
		bingdingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		bingdingDescription.stride = sizeof(Vertex);
		return bingdingDescription;
	}
	static std::array< VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
		std::array< VkVertexInputAttributeDescription, 2> attributeDescriptions{};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[0].location = 0; // 和 shader 中的location相对应
		attributeDescriptions[0].offset = offsetof(Vertex, pos); // 这个是在stddef.h头文件中

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].offset = offsetof(Vertex, color);
		return attributeDescriptions;
	}
};


const std::vector<Vertex> vertices = {
	{{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
	{{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<Vertex> vertices1 = {
	{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
	{{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
	{{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
};

const std::vector<uint16_t> indices = {
	0, 1, 2, 2, 3, 0
};

// @brief: for every object to create a unique model matrix
// @param: radius 环形分布半径， perObjectXY 存储产生的坐标，isUpdate 是否更新
// @ret: void
// birth: created by hermesjang
void generagePerObjectXY(float radius, std::vector<glm::vec3> &perObjectXY) {
	perObjectXY.resize(OBJECT_INSTANCES);
	for (uint32_t i = 0; i < OBJECT_INSTANCES; ++i) {
		perObjectXY[i] = glm::vec3(radius * cos(2 * PI * i / OBJECT_INSTANCES), radius * sin(2 * PI * i / OBJECT_INSTANCES), 0.0f);
	}
}

struct QueueFamilyIndics {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() {
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestoryDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

void* alignedAlloc(size_t size, size_t alignment)
{
	void* data = nullptr;
#if defined(_MSC_VER) || defined(__MINGW32__)
	data = _aligned_malloc(size, alignment);
#else
	int res = posix_memalign(&data, alignment, size);
	if (res != 0)
		data = nullptr;
#endif
	return data;
}

void alignedFree(void* data)
{
#if	defined(_MSC_VER) || defined(__MINGW32__)
	_aligned_free(data);
#else
	free(data);
#endif
}

class HelloTriangleApplication {
public:
	void run() {
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;
	VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
	VkDevice device;
	VkQueue graphicQueue;
	VkQueue presentQueue;

	VkSwapchainKHR swapchain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageView;

	VkDescriptorSetLayout descriptorsetlayout;
	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformMemory;
	std::vector<void*> uniformBuffersMapped;
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;

	VkRenderPass renderPass;
	VkPipelineLayout pipelineLayout; //用来存放uniform的 用于资源绑定的
	VkPipeline pipeline;

	std::vector<VkFramebuffer> swapchainFramebuffer;

	VkCommandPool commandPool;
	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;

	VkCommandBuffer commandBuffer;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexMemory;
	// circle
	VkBuffer circleBuffer;
	VkDeviceMemory circleMemory;
	VkPipelineLayout circlePipelineLayout;
	VkPipeline circlePipeline;
	VkDescriptorSetLayout circleDescriptorSetLayout;
	std::vector<VkBuffer> uniformBuffersCircle;
	std::vector<VkDeviceMemory> uniformMemoryCircle;
	std::vector<void*> uniformBuffersMappedCircle;
	std::vector<VkDescriptorSet> circleDescriptorSets;
	
	struct UniformBufferObject {
		glm::mat4 model;
		glm::mat4 view;
		glm::mat4 proj;
	};

	// uniform buffer and dynamic uniform buffer
	// buffer map data
	struct UniformBuffer {
		VkBuffer view;
		VkBuffer dynamic;
	};
	std::vector<UniformBuffer> dynamicUniformbufffers;
	// memory
	struct UniformMemory {
		VkDeviceMemory viewMemory;
		VkDeviceMemory dyamiceMemory;
	};
	std::vector<UniformMemory> dynamicUniformMemorys;
	// data
	struct UniformMapped {
		void* viewMemoryMaped;
		void* dynamicMemoryMaped;
	};
	std::vector<UniformMapped> dynamicUniformMappeds;

	struct uboVS {
		alignas(16) glm::mat4 view;
		alignas(16) glm::mat4 projection;
	};

	struct UboDataDynamic {
		alignas(8) glm::mat4* model = nullptr;
	};

	std::vector<UboDataDynamic> ubodataDynamics;
	std::vector<glm::vec3> perObjectXY;
	size_t dynamicAlignment;
	VkDescriptorSetLayout dynamicUniformBufferDescriptorSetLayout;
	VkDescriptorPool dynamicUniformBufferDescriptorPool;
	std::vector<VkDescriptorSet> dynamicUniformBufferDescriptorsets;
	VkPipelineLayout dynamicUniformBufferPipelineLayout;
	VkPipeline dynamicUniformBufferPipeline;

	VkSemaphore imageAvailableSemaphore;
	VkSemaphore renderFinishedSemaphore;
	VkFence inFlightFence;

	void initWindow() {
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}

	void initVulkan() {
		createInstance();
		setupDebugMessenger();
		createSurface();
		pickPhysicalDevice();
		createLogicalDevice();
		createSwapChain();
		createImageView();
		createRenderPass();
		//createDescriptorsetLayout(); // descriptor layout
		//createGraphicsPipeline();  // 需要descriptor layout
		//createCircleDescriptorSetLayout();
		createDynamicUniformBufferDescriptorSetLayout(); // descriptor layout
		createDynamicUniformBufferGraphicsPipeline(); // 需要descriptor layout
		createFramebuffer();
		//createDescriptorPool(); // pool 不需要layout
		createDynamicUniformBufferDescriptorPool(); // pool 不需要layout
		createCommandPool();
		//createVertexBuffer();
		//createIndexBuffer();
		createCircleBuffer(); // circles' vertex buffer
		//createCircleUniformBuffers(); 
		createDynamicAndStaticUniformBuffers(); // buffer 也不需要layoutf
		//createUniformBuffers(); // buffer 也不需要layoutf
		createDynamicUniformBufferDescriptorsets(); // 创建sets 需要buffer 需要layout 需要pool
		createCommandBuffers();
		//createCircleDescriptorSets();
		//createDescriptorsets(); // 创建sets 需要buffer 需要layout 需要pool
		createSyncObjects();
	}

	void translateCircle(glm::vec3 location, uint32_t currrentFrame) {
		UniformBufferObject ubo;
		ubo.model = glm::translate(glm::mat4(1.0f), location);
		ubo.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 100.0f);
		ubo.proj[1][1] *= -1;
		//ubo.view = glm::mat4(1.0f);
		//ubo.proj = glm::mat4(1.0f);
		memcpy(uniformBuffersMappedCircle[currentFrame], &ubo, sizeof(ubo));
	}

	void createCircleDescriptorSets() {
		std::vector<VkDescriptorSetLayout> circleLayouts(MAX_FRAMES_IN_FLIGHT, circleDescriptorSetLayout);
		VkDescriptorSetAllocateInfo circleSetAllocateInfo{};
		circleSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		circleSetAllocateInfo.pSetLayouts = circleLayouts.data();
		circleSetAllocateInfo.descriptorPool = descriptorPool;
		circleSetAllocateInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		circleDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &circleSetAllocateInfo, circleDescriptorSets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create circle descriptor sets");
		}
		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			VkDescriptorBufferInfo circleBufferInfo{};
			circleBufferInfo.buffer = uniformBuffersCircle[i];
			circleBufferInfo.offset = 0;
			circleBufferInfo.range = sizeof(UniformBufferObject);

			VkWriteDescriptorSet circleWriteDescriptorSet{};
			circleWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			circleWriteDescriptorSet.descriptorCount = 1;
			circleWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			circleWriteDescriptorSet.dstArrayElement = 0;
			circleWriteDescriptorSet.dstBinding = 0;
			circleWriteDescriptorSet.dstSet = circleDescriptorSets[i];
			circleWriteDescriptorSet.pBufferInfo = &circleBufferInfo;
			circleWriteDescriptorSet.pImageInfo = nullptr;
			circleWriteDescriptorSet.pNext = nullptr;
			circleWriteDescriptorSet.pTexelBufferView = nullptr;
			vkUpdateDescriptorSets(device, 1, &circleWriteDescriptorSet, 0, nullptr);
		}
	}

	void createCircleUniformBuffers() {
		uniformBuffersCircle.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMappedCircle.resize(MAX_FRAMES_IN_FLIGHT);
		uniformMemoryCircle.resize(MAX_FRAMES_IN_FLIGHT);
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);
		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			createBuffer(uniformBuffersCircle[i], uniformMemoryCircle[i], bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
			vkMapMemory(device, uniformMemoryCircle[i], 0, bufferSize, 0, &uniformBuffersMappedCircle[i]);
		}
	}

	// dynamic uniform buffer
	void updateStaticUniformBuffersData(bool update) {
		if (!update) return;
		uboVS ubo;
		ubo.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		ubo.projection = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
		ubo.projection[1][1] *= -1;
		memcpy(dynamicUniformMappeds[currentFrame].viewMemoryMaped, &ubo, sizeof(ubo));
	}

	void updateDynamicUniformBuffersData(bool update) {
		if (!update) return;
		generagePerObjectXY(0.5f, perObjectXY);
		for (uint32_t i = 0; i < perObjectXY.size(); ++i) {
			glm::mat4* matModel = (glm::mat4*)(((uint64_t)ubodataDynamics[currentFrame].model + (i * dynamicAlignment)));
			*matModel = glm::translate(glm::mat4(1.0f), perObjectXY[i]);
		}
		size_t bufferSizeDynamic = OBJECT_INSTANCES * dynamicAlignment;
		memcpy(dynamicUniformMappeds[currentFrame].dynamicMemoryMaped, ubodataDynamics[currentFrame].model, bufferSizeDynamic);
		
		// Flush to make changes visible to the host
		VkMappedMemoryRange memoryRange{};
		memoryRange.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
		memoryRange.memory = dynamicUniformMemorys[currentFrame].dyamiceMemory;
		memoryRange.size = bufferSizeDynamic;
		vkFlushMappedMemoryRanges(device, 1, &memoryRange);
	}

	void createDynamicAndStaticUniformBuffers() {
		//dynamic 获取偏移量
		VkPhysicalDeviceProperties properties;
		vkGetPhysicalDeviceProperties(physicalDevice, &properties);
		size_t minUboAlignment = properties.limits.minUniformBufferOffsetAlignment;
		dynamicAlignment = sizeof(glm::mat4);
		if (minUboAlignment > 0) {
			dynamicAlignment = (dynamicAlignment + minUboAlignment - 1) & ~(minUboAlignment - 1);
		}
		VkDeviceSize bufferSizeDynamic = OBJECT_INSTANCES * dynamicAlignment;
		VkDeviceSize bufferSizeStatic = sizeof(uboVS);

		dynamicUniformbufffers.resize(MAX_FRAMES_IN_FLIGHT);
		dynamicUniformMappeds.resize(MAX_FRAMES_IN_FLIGHT);
		dynamicUniformMemorys.resize(MAX_FRAMES_IN_FLIGHT);
		ubodataDynamics.resize(MAX_FRAMES_IN_FLIGHT);

		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			ubodataDynamics[i].model = (glm::mat4*)alignedAlloc(bufferSizeDynamic, dynamicAlignment);
			createBuffer(dynamicUniformbufffers[i].view, dynamicUniformMemorys[i].viewMemory, bufferSizeStatic, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
			vkMapMemory(device, dynamicUniformMemorys[i].viewMemory, 0, bufferSizeStatic, 0, &dynamicUniformMappeds[i].viewMemoryMaped);
			createBuffer(dynamicUniformbufffers[i].dynamic, dynamicUniformMemorys[i].dyamiceMemory, bufferSizeDynamic, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
			vkMapMemory(device, dynamicUniformMemorys[i].dyamiceMemory, 0, bufferSizeDynamic, 0, &dynamicUniformMappeds[i].dynamicMemoryMaped); // 这儿有问题提
		}
		updateStaticUniformBuffersData(VK_TRUE);
		updateDynamicUniformBuffersData(VK_TRUE);
	}

	void createDynamicUniformBufferDescriptorSetLayout() {
		VkDescriptorSetLayoutBinding uboStaticLayoutBinding{};
		uboStaticLayoutBinding.binding = 0;
		uboStaticLayoutBinding.descriptorCount = 1;
		uboStaticLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		uboStaticLayoutBinding.pImmutableSamplers = nullptr;
		uboStaticLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding uboDynamicLayoutBinding{};
		uboDynamicLayoutBinding.binding = 1;
		uboDynamicLayoutBinding.descriptorCount = 1;
		uboDynamicLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
		uboDynamicLayoutBinding.pImmutableSamplers = nullptr;
		uboDynamicLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding layoutBingding[] = { uboStaticLayoutBinding , uboDynamicLayoutBinding };

		VkDescriptorSetLayoutCreateInfo dynamicUniformBufferDescriptorSetLayoutCreateInfo{};
		dynamicUniformBufferDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		dynamicUniformBufferDescriptorSetLayoutCreateInfo.bindingCount = 2;
		dynamicUniformBufferDescriptorSetLayoutCreateInfo.pBindings = layoutBingding;
		if (vkCreateDescriptorSetLayout(device, &dynamicUniformBufferDescriptorSetLayoutCreateInfo, nullptr, &dynamicUniformBufferDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create dynamic uniform buffer descriptorsetlayouts");
		}
	}

	void createDynamicUniformBufferDescriptorPool() {
		VkDescriptorPoolSize staticPoolSize{};
		staticPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		staticPoolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		VkDescriptorPoolSize dynamicPoolSize{};
		dynamicPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
		dynamicPoolSize.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

		std::vector<VkDescriptorPoolSize> poolSizes = { staticPoolSize , dynamicPoolSize };

		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
		descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptorPoolCreateInfo.poolSizeCount = poolSizes.size();
		descriptorPoolCreateInfo.pPoolSizes = poolSizes.data();
		descriptorPoolCreateInfo.maxSets = 2 * static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT); // 这是池子里最大的数？
		if (vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &dynamicUniformBufferDescriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptorpool");
		}
	}

	void createDynamicUniformBufferDescriptorsets() {
		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, dynamicUniformBufferDescriptorSetLayout);
		VkDescriptorSetAllocateInfo setAllocateInfo{};
		setAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		setAllocateInfo.descriptorPool = dynamicUniformBufferDescriptorPool;
		setAllocateInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		setAllocateInfo.pSetLayouts = layouts.data();
		dynamicUniformBufferDescriptorsets.resize(MAX_FRAMES_IN_FLIGHT);
		if (vkAllocateDescriptorSets(device, &setAllocateInfo, dynamicUniformBufferDescriptorsets.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to create dynamic uniform buffer descriptor sets");
		}

		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			VkDescriptorBufferInfo bufferInfo = {};
			bufferInfo.buffer = dynamicUniformbufffers[i].view;
			bufferInfo.offset = 0;
			bufferInfo.range = sizeof(uboVS);

			VkDescriptorBufferInfo dynamicBufferInfo = {};
			dynamicBufferInfo.buffer = dynamicUniformbufffers[i].dynamic;
			dynamicBufferInfo.offset = 0;
			dynamicBufferInfo.range = dynamicAlignment;

			VkWriteDescriptorSet staticWriteDescriptor{};
			staticWriteDescriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			staticWriteDescriptor.dstSet = dynamicUniformBufferDescriptorsets[i];
			staticWriteDescriptor.dstBinding = 0;
			staticWriteDescriptor.dstArrayElement = 0;
			staticWriteDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			staticWriteDescriptor.descriptorCount = 1;
			staticWriteDescriptor.pBufferInfo = &bufferInfo;
			staticWriteDescriptor.pImageInfo = nullptr;
			staticWriteDescriptor.pTexelBufferView = nullptr;

			VkWriteDescriptorSet dynamicWriteDescriptor{};
			dynamicWriteDescriptor.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			dynamicWriteDescriptor.dstSet = dynamicUniformBufferDescriptorsets[i];
			dynamicWriteDescriptor.dstBinding = 1;
			dynamicWriteDescriptor.dstArrayElement = 0;
			dynamicWriteDescriptor.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
			dynamicWriteDescriptor.descriptorCount = 1;
			dynamicWriteDescriptor.pBufferInfo = &dynamicBufferInfo;
			dynamicWriteDescriptor.pTexelBufferView = nullptr;
			std::vector<VkWriteDescriptorSet> writeDescriptorsets = { staticWriteDescriptor , dynamicWriteDescriptor };
			vkUpdateDescriptorSets(device, 2, writeDescriptorsets.data(), 0, nullptr); // 填充buffer内容 我是傻逼 这儿肯定是2啊
		}
	}

	void createDynamicUniformBufferGraphicsPipeline() {
		// fix stage
		auto vertShaderCode = readFile("circlevert.spv");
		auto fragShaderCode = readFile("circlefrag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		// fix function
		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicSateCreateInfo{};
		dynamicSateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicSateCreateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicSateCreateInfo.pDynamicStates = dynamicStates.data();

		VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo{};
		vertexInputCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		auto bingDingDescription = CirclePoint::getBindingDescription();
		auto attributeDescription = CirclePoint::getAttributeDescriptions();

		vertexInputCreateInfo.vertexBindingDescriptionCount = 1;
		vertexInputCreateInfo.pVertexBindingDescriptions = &bingDingDescription;
		vertexInputCreateInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescription.size());
		vertexInputCreateInfo.pVertexAttributeDescriptions = attributeDescription.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewPort{};
		viewPort.x = 0.0f;
		viewPort.y = 0.0f;
		viewPort.width = (float)swapChainExtent.width;
		viewPort.height = (float)swapChainExtent.height;
		viewPort.minDepth = 0.0f;
		viewPort.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.extent = swapChainExtent;
		scissor.offset = { 0, 0 };

		VkPipelineViewportStateCreateInfo viewPortState{};
		viewPortState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewPortState.viewportCount = 1;
		viewPortState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.lineWidth = 1.0f;

		rasterizer.cullMode = VK_CULL_MODE_NONE;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

		VkPipelineColorBlendStateCreateInfo colorBlend{};
		colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlend.logicOpEnable = VK_FALSE;
		colorBlend.logicOp = VK_LOGIC_OP_COPY;
		colorBlend.attachmentCount = 1;
		colorBlend.pAttachments = &colorBlendAttachment;
		colorBlend.blendConstants[0] = 0.0f; // rgba
		colorBlend.blendConstants[1] = 0.0f;
		colorBlend.blendConstants[2] = 0.0f;
		colorBlend.blendConstants[3] = 0.0f;

		VkPipelineLayoutCreateInfo layoutCreateInfo{};
		layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutCreateInfo.setLayoutCount = 1; 
		layoutCreateInfo.pSetLayouts = &dynamicUniformBufferDescriptorSetLayout;
		layoutCreateInfo.pushConstantRangeCount = 0;
		layoutCreateInfo.pPushConstantRanges = nullptr;
		if (vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &dynamicUniformBufferPipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create dynamic uniform buffer layout");
		}

		VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo{};
		graphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		graphicsPipelineCreateInfo.stageCount = 2;
		graphicsPipelineCreateInfo.pStages = shaderStages;

		graphicsPipelineCreateInfo.pDynamicState = &dynamicSateCreateInfo;
		graphicsPipelineCreateInfo.pVertexInputState = &vertexInputCreateInfo;
		graphicsPipelineCreateInfo.pInputAssemblyState = &inputAssembly;
		graphicsPipelineCreateInfo.pViewportState = &viewPortState;
		graphicsPipelineCreateInfo.pRasterizationState = &rasterizer;
		graphicsPipelineCreateInfo.pMultisampleState = &multisampling;
		graphicsPipelineCreateInfo.pDepthStencilState = nullptr;
		graphicsPipelineCreateInfo.pColorBlendState = &colorBlend;
		graphicsPipelineCreateInfo.layout = dynamicUniformBufferPipelineLayout;

		graphicsPipelineCreateInfo.renderPass = renderPass;
		graphicsPipelineCreateInfo.subpass = 0;

		graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
		graphicsPipelineCreateInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineCreateInfo, nullptr, &dynamicUniformBufferPipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline");
		}
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
	}

	// dynamic uniform buffer

	void createCircleDescriptorSetLayout() {
		VkDescriptorSetLayoutBinding circleDescriptorSetLayoutBinding{};
		circleDescriptorSetLayoutBinding.binding = 0;
		circleDescriptorSetLayoutBinding.descriptorCount = 1;
		circleDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		circleDescriptorSetLayoutBinding.pImmutableSamplers = nullptr;
		circleDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding uboDynamicLayoutBinding{};
		uboDynamicLayoutBinding.binding = 1;
		uboDynamicLayoutBinding.descriptorCount = 1;
		uboDynamicLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
		uboDynamicLayoutBinding.pImmutableSamplers = nullptr;
		uboDynamicLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding layoutBingding[] = {circleDescriptorSetLayoutBinding , uboDynamicLayoutBinding};

		VkDescriptorSetLayoutCreateInfo circleDescriptorSetLayoutCreateInfo{};
		circleDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		circleDescriptorSetLayoutCreateInfo.bindingCount = 2;
		circleDescriptorSetLayoutCreateInfo.pBindings = layoutBingding;
		if (vkCreateDescriptorSetLayout(device, &circleDescriptorSetLayoutCreateInfo, nullptr, &circleDescriptorSetLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create circle descriptorsetlayouts");
		}
	}

	void createDescriptorPool() {
		std::vector<VkDescriptorPoolSize> poolSizes{};
		poolSizes.resize(2);
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
		poolSizes[1].descriptorCount = static_cast<uint32_t>(swapChainImages.size());

		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
		descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptorPoolCreateInfo.poolSizeCount = 2;
		descriptorPoolCreateInfo.pPoolSizes = poolSizes.data();
		descriptorPoolCreateInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT); // 这是池子里最大的数？
		if (vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptorpool");
		}
	}

	void createCirclePipeline() {
		// fix stage
		auto vertShaderCode = readFile("circlevert.spv");
		auto fragShaderCode = readFile("circlefrag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		// fix function
		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicSateCreateInfo{};
		dynamicSateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicSateCreateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicSateCreateInfo.pDynamicStates = dynamicStates.data();

		VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo{};
		vertexInputCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		auto bingDingDescription = CirclePoint::getBindingDescription();
		auto attributeDescription = CirclePoint::getAttributeDescriptions();

		vertexInputCreateInfo.vertexBindingDescriptionCount = 1;
		vertexInputCreateInfo.pVertexBindingDescriptions = &bingDingDescription;
		vertexInputCreateInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescription.size());
		vertexInputCreateInfo.pVertexAttributeDescriptions = attributeDescription.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewPort{};
		viewPort.x = 0.0f;
		viewPort.y = 0.0f;
		viewPort.width = (float)swapChainExtent.width;
		viewPort.height = (float)swapChainExtent.height;
		viewPort.minDepth = 0.0f;
		viewPort.maxDepth = 1.0f;
		
		VkRect2D scissor{};
		scissor.extent = swapChainExtent;
		scissor.offset = {0, 0};

		VkPipelineViewportStateCreateInfo viewPortState{};
		viewPortState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewPortState.viewportCount = 1;
		viewPortState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.lineWidth = 1.0f;

		rasterizer.cullMode = VK_CULL_MODE_NONE;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

		VkPipelineColorBlendStateCreateInfo colorBlend{};
		colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlend.logicOpEnable = VK_FALSE;
		colorBlend.logicOp = VK_LOGIC_OP_COPY;
		colorBlend.attachmentCount = 1;
		colorBlend.pAttachments = &colorBlendAttachment;
		colorBlend.blendConstants[0] = 0.0f; // rgba
		colorBlend.blendConstants[1] = 0.0f;
		colorBlend.blendConstants[2] = 0.0f;
		colorBlend.blendConstants[3] = 0.0f;

		VkPipelineLayoutCreateInfo layoutCreateInfo{};
		layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutCreateInfo.setLayoutCount = 1;
		layoutCreateInfo.pSetLayouts = &circleDescriptorSetLayout;
		layoutCreateInfo.pushConstantRangeCount = 0;
		layoutCreateInfo.pPushConstantRanges = nullptr;
		if (vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &circlePipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create layout");
		}

		VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo{};
		graphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		graphicsPipelineCreateInfo.stageCount = 2;
		graphicsPipelineCreateInfo.pStages = shaderStages;

		graphicsPipelineCreateInfo.pDynamicState = &dynamicSateCreateInfo;
		graphicsPipelineCreateInfo.pVertexInputState = &vertexInputCreateInfo;
		graphicsPipelineCreateInfo.pInputAssemblyState = &inputAssembly;
		graphicsPipelineCreateInfo.pViewportState = &viewPortState;
		graphicsPipelineCreateInfo.pRasterizationState = &rasterizer;
		graphicsPipelineCreateInfo.pMultisampleState = &multisampling;
		graphicsPipelineCreateInfo.pDepthStencilState = nullptr;
		graphicsPipelineCreateInfo.pColorBlendState = &colorBlend;
		graphicsPipelineCreateInfo.layout = circlePipelineLayout;

		graphicsPipelineCreateInfo.renderPass = renderPass;
		graphicsPipelineCreateInfo.subpass = 0;

		graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
		graphicsPipelineCreateInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineCreateInfo, nullptr, &circlePipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline");
		}
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
	}

	void createCircleBuffer() {
		generateCircle(0.1f, 20, circlePointSet);
		VkDeviceSize bufferSize = sizeof(circlePointSet[0]) * circlePointSet.size();
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingMemory;
		createBuffer(stagingBuffer, stagingMemory, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		void* data;
		vkMapMemory(device, stagingMemory, 0, bufferSize, 0, &data);
		memcpy(data, circlePointSet.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingMemory);

		createBuffer(circleBuffer, circleMemory, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		copyBuffer(stagingBuffer, circleBuffer, bufferSize);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingMemory, nullptr);
	}

	void updateUniformData(uint32_t currentframe) {
		static auto startTime = std::chrono::high_resolution_clock::now();
		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
		UniformBufferObject ubo;
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width/(float)swapChainExtent.height, 0.1f, 10.0f);
		ubo.proj[1][1] *= -1;
		memcpy(uniformBuffersMapped[currentFrame], &ubo, sizeof(ubo));
	}

	void createUniformBuffers() {
		VkDeviceSize bufferSize = sizeof(UniformBufferObject);
		uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		uniformMemory.resize(MAX_FRAMES_IN_FLIGHT);
		uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			createBuffer(uniformBuffers[i], uniformMemory[i], bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
			vkMapMemory(device, uniformMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
		}
	}

	void createDescriptorsetLayout() {
		VkDescriptorSetLayoutBinding layoutBinding;
		layoutBinding.binding = 0; // 和vertex shader中的binding对应
		layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		layoutBinding.descriptorCount = 1; // 因为只在一个unifrombuffer中，因此只有一个
		layoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
		layoutBinding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutCreateInfo setLayoutCreateInfo{};
		setLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		setLayoutCreateInfo.bindingCount = 1;
		setLayoutCreateInfo.pBindings = &layoutBinding;

		if (vkCreateDescriptorSetLayout(device, &setLayoutCreateInfo, nullptr, &descriptorsetlayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create descriptorsetlayout");
		}
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags property) {
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
		for (int i = 0; i < memProperties.memoryTypeCount; ++i) {
			if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & property) == property) {
				return i; // 所给的内存块的类型是否满足要求
			}	
		}
		throw std::runtime_error("failed to find suitable memory");
	}

	void createBuffer(VkBuffer &buffer, VkDeviceMemory &memory, VkDeviceSize size, VkBufferUsageFlags useage, VkMemoryPropertyFlags properties) {
		VkBufferCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		createInfo.usage = useage;
		createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE; // 只会被graphic queue使用
		createInfo.size = size; // 这是你要的大小
		vkCreateBuffer(device, &createInfo, nullptr, &buffer);

		VkMemoryRequirements memRequirement{};
		vkGetBufferMemoryRequirements(device, buffer, &memRequirement);

		VkMemoryAllocateInfo allocateInfo{};
		allocateInfo.allocationSize = memRequirement.size;
		allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocateInfo.memoryTypeIndex = findMemoryType(memRequirement.memoryTypeBits, properties);
		if (vkAllocateMemory(device, &allocateInfo, nullptr, &memory) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate memory");
		}
		vkBindBufferMemory(device, buffer, memory, 0);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
		VkCommandBufferAllocateInfo bufferAllocateInfo{};
		bufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		bufferAllocateInfo.commandBufferCount = 1;
		bufferAllocateInfo.commandPool = commandPool;
		bufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		VkCommandBuffer commandBuffer;
		if (vkAllocateCommandBuffers(device, &bufferAllocateInfo, &commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to create copycommandbuffer");
		}

		VkCommandBufferBeginInfo commandBufferBeginInfo{};
		commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);

		// 其实只需要这一段话，其他的都是固定的，重复的
		VkBufferCopy copyRegin{};
		copyRegin.size = size;
		copyRegin.srcOffset = 0;
		copyRegin.dstOffset = 0;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegin);

		vkEndCommandBuffer(commandBuffer);
		
		VkSubmitInfo submitInfo{};
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		vkQueueSubmit(graphicQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicQueue);
		vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
	}

	void createIndexBuffer() {
		VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingMemory;
		createBuffer(stagingBuffer, stagingMemory, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		void* data;
		vkMapMemory(device, stagingMemory, 0, bufferSize, 0, &data);
		memcpy(data, indices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingMemory);

		createBuffer(indexBuffer, indexMemory, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		copyBuffer(stagingBuffer, indexBuffer, bufferSize);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingMemory, nullptr);
	}

	void createVertexBuffer() {
		VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
		VkBuffer stagingBuffer;
		VkDeviceMemory stagingMemory;
		createBuffer(stagingBuffer, stagingMemory, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

		void* data;
		vkMapMemory(device, stagingMemory, 0, bufferSize, 0, &data);
		memcpy(data, vertices.data(), (size_t)bufferSize);
		vkUnmapMemory(device, stagingMemory);

		createBuffer(vertexBuffer, vertexMemory, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
		copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
		vkDestroyBuffer(device, stagingBuffer, nullptr);
		vkFreeMemory(device, stagingMemory, nullptr);
	}

	void createSyncObjects() {
		VkSemaphoreCreateInfo semaphoreCreateInfo{};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceCreateInfo{};
		fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			if (vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceCreateInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS
				) {
				throw std::runtime_error("failed to create semaphore or/and fence");
			}
		}
	}

	void drawFrame() {
		// 等前一frame完成
		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		// 从swapchain 中获取image
		uint32_t imgeIndex;
		vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imgeIndex);

		// record command buffer
		// updateUniformData(currentFrame);
		// translateCircle(glm::vec3(0.8f, 0.8f, 0.0f), currentFrame);
		updateStaticUniformBuffersData(VK_TRUE);
		updateDynamicUniformBuffersData(VK_TRUE);
		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imgeIndex);

		// submit the command buffer
		VkSemaphore waitSemaphore[] = {imageAvailableSemaphores[currentFrame]};
		VkSemaphore signalSemaphore[] = {renderFinishedSemaphores[currentFrame]};
		VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
		submitInfo.pSignalSemaphores = signalSemaphore;
		submitInfo.pWaitDstStageMask = waitStages;
		submitInfo.pWaitSemaphores = waitSemaphore;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.waitSemaphoreCount = 1;
		if (vkQueueSubmit(graphicQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
			throw std::runtime_error("failed to submit draw command buffer");
		}

		// subpass dependency see create render pass

		//presentation
		VkPresentInfoKHR presentInfo{};
		VkSwapchainKHR swapchains[] = {swapchain};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.pImageIndices = &imgeIndex;
		presentInfo.pSwapchains = swapchains;
		presentInfo.pWaitSemaphores = signalSemaphore;
		presentInfo.swapchainCount = 1;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pResults = nullptr;
		vkQueuePresentKHR(presentQueue, &presentInfo);
		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void createCommandBuffers() {
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = commandPool;
		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
			throw std::runtime_error("failed to allocate command buffer");
		}
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
		VkCommandBufferBeginInfo commandBufferBeginInfo{};
		commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		commandBufferBeginInfo.flags = 0;
		commandBufferBeginInfo.pInheritanceInfo = nullptr;
		if (vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo) != VK_SUCCESS) {
			throw std::runtime_error("failed to begin command buffer");
		}

		VkClearValue clearColor = { {{0.0f, 0.0f, 1.0f, 1.0f}} };
		VkRenderPassBeginInfo renderPassBeginInfo{};
		renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassBeginInfo.clearValueCount = 1;
		renderPassBeginInfo.pClearValues = &clearColor;
		renderPassBeginInfo.framebuffer = swapchainFramebuffer[imageIndex];
		renderPassBeginInfo.renderArea.offset = {0, 0};
		renderPassBeginInfo.renderArea.extent = swapChainExtent;
		renderPassBeginInfo.renderPass = renderPass;
		vkCmdBeginRenderPass(commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

		//vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, dynamicUniformBufferPipeline);

		VkViewport viewPort{};
		viewPort.x = 0.0f;
		viewPort.y = 0.0f;
		viewPort.width = static_cast<uint32_t>(swapChainExtent.width);
		viewPort.height = static_cast<uint32_t>(swapChainExtent.height);
		viewPort.minDepth = 0.0f;
		viewPort.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewPort);

		VkRect2D scissor{};
		scissor.offset = {0, 0}; // int32_t
		scissor.extent = swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		//VkBuffer vertexBuffers[] = { vertexBuffer };
		VkBuffer vertexBuffers[] = { circleBuffer };
		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);
		// vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, circlePipelineLayout, 0, 1, &circleDescriptorSets[currentFrame], 0, nullptr);
		// vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);
		// vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
		for (uint32_t j = 0; j < OBJECT_INSTANCES; ++j) {
			uint32_t dynamicOffset = j * static_cast<uint32_t>(dynamicAlignment);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, dynamicUniformBufferPipelineLayout, 0, 1, &dynamicUniformBufferDescriptorsets[currentFrame], 1, &dynamicOffset);
			vkCmdDraw(commandBuffer, static_cast<uint32_t>(circlePointSet.size()), 1, 0, 0);
		}
		//vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
		vkCmdEndRenderPass(commandBuffer);
		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
			throw std::runtime_error("failed to record command buffer");
		}
	}

	void createCommandPool() {
		QueueFamilyIndics queueFamilyIndics = findQueueFamilies(physicalDevice);
		VkCommandPoolCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		createInfo.queueFamilyIndex = queueFamilyIndics.graphicsFamily.value();
		if (vkCreateCommandPool(device, &createInfo, nullptr, &commandPool) != VK_SUCCESS) {
			throw std::runtime_error("failed to create command pool");
		}
	}

	void createFramebuffer() {
		swapchainFramebuffer.resize(swapChainImageView.size());
		for (int i = 0; i < swapchainFramebuffer.size(); ++i) {
			VkImageView attachmet[] = {
				swapChainImageView[i]
			};
			VkFramebufferCreateInfo frambufferCreateInfo{};
			frambufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			frambufferCreateInfo.attachmentCount = 1;
			frambufferCreateInfo.pAttachments = attachmet;
			frambufferCreateInfo.renderPass = renderPass;
			frambufferCreateInfo.width = swapChainExtent.width;
			frambufferCreateInfo.height = swapChainExtent.height;
			frambufferCreateInfo.layers = 1;
			if (vkCreateFramebuffer(device, &frambufferCreateInfo, nullptr, &swapchainFramebuffer[i]) != VK_SUCCESS) {
				throw std::runtime_error("failed to create framebuffer");
			}
		}
	}

	void createRenderPass() {
		VkAttachmentDescription colorAttachmentDescription{};
		colorAttachmentDescription.format = swapChainImageFormat;
		colorAttachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
		colorAttachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		colorAttachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; // do nothing in stencil buffer
		colorAttachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		// attachment 就是要渲染出的结果是啥 render target 就是一个缓冲区，用来保存渲染结果的
		VkAttachmentReference colorAttachmentReference{};
		colorAttachmentReference.attachment = 0;
		colorAttachmentReference.layout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subPass{};
		subPass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subPass.colorAttachmentCount = 1;
		subPass.pColorAttachments = &colorAttachmentReference;

		VkSubpassDependency subpassDependency{};
		subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		subpassDependency.dstSubpass = 0;
		subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		subpassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		subpassDependency.srcAccessMask = 0;
		subpassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		VkRenderPassCreateInfo renderpassCreateInfo{};
		renderpassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderpassCreateInfo.attachmentCount = 1;
		renderpassCreateInfo.subpassCount = 1;
		renderpassCreateInfo.pAttachments = &colorAttachmentDescription;
		renderpassCreateInfo.pSubpasses = &subPass;
		renderpassCreateInfo.dependencyCount = 1;
		renderpassCreateInfo.pDependencies = &subpassDependency;

		if (vkCreateRenderPass(device, &renderpassCreateInfo, nullptr, &renderPass) != VK_SUCCESS) {
			throw std::runtime_error("failed to create renderpass");
		}

	}

	VkShaderModule createShaderModule(const std::vector<char>& code) {
		VkShaderModuleCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
			throw std::runtime_error("failed to create shader module");
		}
		return shaderModule;
	}

	static std::vector<char> readFile(const std::string & filename) {
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open()) {
			throw std::runtime_error("failed to open file!");
		}
		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();
		return buffer;
	}

	void createGraphicsPipeline() {
		auto vertShaderCode = readFile("vert.spv");
		auto fragShaderCode = readFile("frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";

		VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

		// fix function
		std::vector<VkDynamicState> dynamicStates = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicSateCreateInfo{};
		dynamicSateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicSateCreateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicSateCreateInfo.pDynamicStates = dynamicStates.data();

		VkPipelineVertexInputStateCreateInfo vertexInputCreateInfo{};
		vertexInputCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		auto bingDingDescription = Vertex::getBingdingDescrition();
		auto attributeDescription = Vertex::getAttributeDescriptions();

		vertexInputCreateInfo.vertexBindingDescriptionCount = 1;
		vertexInputCreateInfo.pVertexBindingDescriptions = &bingDingDescription;
		vertexInputCreateInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescription.size());
		vertexInputCreateInfo.pVertexAttributeDescriptions = attributeDescription.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewPort{};
		viewPort.x = 0.0f;
		viewPort.y = 0.0f;
		viewPort.width = (float)swapChainExtent.width;
		viewPort.height = (float)swapChainExtent.height;
		viewPort.minDepth = 0.0f;
		viewPort.maxDepth = 1.0f;
		
		VkRect2D scissor{};
		scissor.extent = swapChainExtent;
		scissor.offset = {0, 0};

		VkPipelineViewportStateCreateInfo viewPortState{};
		viewPortState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewPortState.viewportCount = 1;
		viewPortState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		//rasterizer.polygonMode = VK_POLYGON_MODE_POINT;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.lineWidth = 1.0f;

		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_FALSE;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

		VkPipelineColorBlendStateCreateInfo colorBlend{};
		colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlend.logicOpEnable = VK_FALSE;
		colorBlend.logicOp = VK_LOGIC_OP_COPY;
		colorBlend.attachmentCount = 1;
		colorBlend.pAttachments = &colorBlendAttachment;
		colorBlend.blendConstants[0] = 0.0f; // rgba
		colorBlend.blendConstants[1] = 0.0f;
		colorBlend.blendConstants[2] = 0.0f;
		colorBlend.blendConstants[3] = 0.0f;

		VkPipelineLayoutCreateInfo layoutCreateInfo{};
		layoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutCreateInfo.setLayoutCount = 1;
		layoutCreateInfo.pSetLayouts = &descriptorsetlayout;
		layoutCreateInfo.pushConstantRangeCount = 0;
		layoutCreateInfo.pPushConstantRanges = nullptr;
		if (vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
			throw std::runtime_error("failed to create layout");
		}

		VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo{};
		graphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		graphicsPipelineCreateInfo.stageCount = 2;
		graphicsPipelineCreateInfo.pStages = shaderStages;

		graphicsPipelineCreateInfo.pDynamicState = &dynamicSateCreateInfo;
		graphicsPipelineCreateInfo.pVertexInputState = &vertexInputCreateInfo;
		graphicsPipelineCreateInfo.pInputAssemblyState = &inputAssembly;
		graphicsPipelineCreateInfo.pViewportState = &viewPortState;
		graphicsPipelineCreateInfo.pRasterizationState = &rasterizer;
		graphicsPipelineCreateInfo.pMultisampleState = &multisampling;
		graphicsPipelineCreateInfo.pDepthStencilState = nullptr;
		graphicsPipelineCreateInfo.pColorBlendState = &colorBlend;
		graphicsPipelineCreateInfo.layout = pipelineLayout;

		graphicsPipelineCreateInfo.renderPass = renderPass;
		graphicsPipelineCreateInfo.subpass = 0;

		graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
		graphicsPipelineCreateInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineCreateInfo, nullptr, &pipeline) != VK_SUCCESS) {
			throw std::runtime_error("failed to create graphics pipeline");
		}
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
		vkDestroyShaderModule(device, fragShaderModule, nullptr);
	}

	void createImageView() {
		swapChainImageView.resize(swapChainImages.size());
		for (int i = 0; i < swapChainImages.size(); ++i) {
			VkImageViewCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			createInfo.image = swapChainImages[i];

			createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			createInfo.format = swapChainImageFormat;

			createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

			createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			createInfo.subresourceRange.baseMipLevel = 0;
			createInfo.subresourceRange.levelCount = 1;
			createInfo.subresourceRange.baseArrayLayer = 0;
			createInfo.subresourceRange.layerCount = 1;
			if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageView[i])) {
				throw std::runtime_error("failed to create image view");
			}
		}
	}

	// swap chain
	void createSwapChain() {
		SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
		if (swapChainSupport.capabilities.maxImageCount > 0 && swapChainSupport.capabilities.maxImageCount < imageCount) {
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;
		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		QueueFamilyIndics indices = findQueueFamilies(physicalDevice);
		uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };
		if (indices.graphicsFamily != indices.presentFamily) {
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = queueFamilyIndices;
		}
		else {
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices = nullptr;
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;
		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapchain)!=VK_SUCCESS) {
			throw std::runtime_error("failed to create swap chain!");
		}
		uint32_t iamgeCount = 0;
		vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapChainImages.data());
		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;

	}
	// window surface
	void createSurface() {
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
	}

	// physicaldevice
	void pickPhysicalDevice() {
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
		if (deviceCount == 0) {
			throw std::runtime_error("failed to find GPUs with Vulkan support!");
		}
		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices) {
			if (isDeviceSuitable(device)) {
				physicalDevice = device;
			}
		}
	}

	SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount = 0;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
		if (formatCount != 0) {
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t modeCount = 0;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &modeCount, nullptr);
		if (modeCount != 0) {
			details.presentModes.resize(modeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &modeCount, details.presentModes.data());
		}
		return details;
	}

	bool isDeviceSuitable(VkPhysicalDevice device) {
		VkPhysicalDeviceProperties deviceProperties;
		VkPhysicalDeviceFeatures deviceFeatures;
		vkGetPhysicalDeviceProperties(device, &deviceProperties);
		vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
		QueueFamilyIndics indices = findQueueFamilies(device);
		bool extensionsSupport = checkDeviceExtensionSupport(device);
		bool swapChainAdequate = false;
		if (extensionsSupport) {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
			swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
		}
		return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
			swapChainAdequate &&
			extensionsSupport && 
			VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && 
			deviceFeatures.geometryShader && 
			indices.isComplete();
	}

	bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
		uint32_t extensionCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
		for (const VkExtensionProperties& extension : availableExtensions) {
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	// more settings
	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
		for (const auto& availableFormat : availableFormats) {
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR) {
				return availableFormat;
			}
		}
		return availableFormats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
		for (const auto& availablePresentMode : availablePresentModes) {
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
				return availablePresentMode;
			}
		}
		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
		if (capabilities.currentExtent.height != std::numeric_limits<uint32_t>::max()) {
			return capabilities.currentExtent;
		}
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		VkExtent2D actualExtent = {
			static_cast<uint32_t>(width),
			static_cast<uint32_t>(height)
		};
		actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
		actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
		return actualExtent;
	}

	// queue families
	QueueFamilyIndics findQueueFamilies(VkPhysicalDevice device) {
		QueueFamilyIndics indices;
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

		int i = 0;
		for (const auto& queueFamily : queueFamilies) {
			if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
				indices.graphicsFamily = i;
			}
			VkBool32 presentSupport = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
			if (presentSupport) {
				indices.presentFamily = i;
			}
			if (indices.isComplete()) {
				break;
			}
			i++;
		}
		return indices;
	}

	// physicaldevice end
	// logical device
	void createLogicalDevice() {
		// queue
		QueueFamilyIndics indices = findQueueFamilies(physicalDevice);

		std::vector<VkDeviceQueueCreateInfo> quequeCreateInfos;

		std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

		float queuePriority = 1.0f;

		for (auto queueFamily : uniqueQueueFamilies) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamily;
			queueCreateInfo.queueCount = 1;
			queueCreateInfo.pQueuePriorities = &queuePriority;
			quequeCreateInfos.push_back(queueCreateInfo);
		}

		// VkDeviceQueueCreateInfo queueCreateInfo{};
		/*queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
		queueCreateInfo.queueCount = 1;*/


		//queueCreateInfo.pQueuePriorities = &queuePriority;

		//feature
		VkPhysicalDeviceFeatures deviceFeatures{};

		// device create
		VkDeviceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		createInfo.pQueueCreateInfos = quequeCreateInfos.data();
		createInfo.queueCreateInfoCount = static_cast<uint32_t>(uniqueQueueFamilies.size());
		createInfo.pEnabledFeatures = &deviceFeatures;

		createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		createInfo.ppEnabledExtensionNames = deviceExtensions.data();
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			createInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device");
		}
		vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicQueue);
		vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
	}

	// logical device end

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData) {

		std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
		createInfo.pUserData = nullptr;
	}

	void setupDebugMessenger() {
		if (!enableValidationLayers) return;
		VkDebugUtilsMessengerCreateInfoEXT createInfo{};
		populateDebugMessengerCreateInfo(createInfo);
		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
			throw std::runtime_error("failed to set up debug messenger!");
		}
	}

	std::vector<const char*> getRequiredExtensions() {
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

		return extensions;
	}

	bool checkValidationLayerSupport() {
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers) {
			bool layerFound = false;
			for (const auto& layerProperties : availableLayers) {
				if (strcmp(layerName, layerProperties.layerName) == 0) {
					layerFound = true;
					break;
				}
			}
			if (!layerFound) {
				return false;
			}
		}
		return true;
	}

	void createInstance() {
		// application info, tell driver detail about your application
		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Hello Triangle";
		appInfo.pEngineName = "No Engine";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_0;

		// make instance
		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;

		// extension
		/*uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions;
		glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		createInfo.enabledExtensionCount = glfwExtensionCount;
		createInfo.ppEnabledExtensionNames = glfwExtensions;*/
		auto extensions = getRequiredExtensions();
		createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		createInfo.ppEnabledExtensionNames = extensions.data();

		/*uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
		std::vector<VkExtensionProperties> extensions(extensionCount);
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());*/

		// validation
		if (enableValidationLayers && !checkValidationLayerSupport()) {
			throw std::runtime_error("validation layers requested, but not available!");
		}
		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers) {
			createInfo.enabledLayerCount = validationLayers.size();
			createInfo.ppEnabledLayerNames = validationLayers.data();
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else {
			createInfo.enabledLayerCount = 0;
			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
			throw std::runtime_error("failed to create instance");
		}
	}

	void mainLoop() {
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			drawFrame();
		}
		vkDeviceWaitIdle(device);
	}

	void cleanup() {
		if (enableValidationLayers) {
			DestoryDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}
		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
			vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
		}

		/*for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			vkDestroyBuffer(device, uniformBuffers[i], nullptr);
			vkFreeMemory(device, uniformMemory[i], nullptr);
		}*/

		//vkDestroyDescriptorSetLayout(device, descriptorsetlayout, nullptr);
		
		/*for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			vkDestroyBuffer(device, uniformBuffersCircle[i], nullptr);
			vkFreeMemory(device, uniformMemoryCircle[i], nullptr);
		}*/
		/*vkDestroyDescriptorSetLayout(device, circleDescriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);*/
		
		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
			vkDestroyBuffer(device, dynamicUniformbufffers[i].view, nullptr);
			vkDestroyBuffer(device, dynamicUniformbufffers[i].dynamic, nullptr);
			vkFreeMemory(device, dynamicUniformMemorys[i].viewMemory, nullptr);
			vkFreeMemory(device, dynamicUniformMemorys[i].dyamiceMemory, nullptr);
			alignedFree(ubodataDynamics[i].model);
		}
		vkDestroyDescriptorSetLayout(device, dynamicUniformBufferDescriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(device, dynamicUniformBufferDescriptorPool, nullptr);

		vkDestroyBuffer(device, circleBuffer, nullptr);
		vkFreeMemory(device, circleMemory, nullptr);

		//vkDestroyBuffer(device, indexBuffer, nullptr);
		//vkFreeMemory(device, indexMemory, nullptr);

		//vkDestroyBuffer(device, vertexBuffer, nullptr);
		//vkFreeMemory(device, vertexMemory, nullptr);
		vkDestroyCommandPool(device, commandPool, nullptr);

		//vkDestroyPipeline(device, pipeline, nullptr);
		//vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		/*vkDestroyPipeline(device, circlePipeline, nullptr);
		vkDestroyPipelineLayout(device, circlePipelineLayout, nullptr);*/
		vkDestroyPipeline(device, dynamicUniformBufferPipeline, nullptr);
		vkDestroyPipelineLayout(device, dynamicUniformBufferPipelineLayout, nullptr);
		vkDestroyRenderPass(device, renderPass, nullptr);

		for (auto framBuffer : swapchainFramebuffer) {
			vkDestroyFramebuffer(device, framBuffer, nullptr);
		}

		for (auto imageView : swapChainImageView) {
			vkDestroyImageView(device, imageView, nullptr);
		}

		vkDestroySwapchainKHR(device, swapchain, nullptr);
		vkDestroyDevice(device, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);
		glfwTerminate();
	}
};

int main() {
	HelloTriangleApplication app;
	try {
		app.run();
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}