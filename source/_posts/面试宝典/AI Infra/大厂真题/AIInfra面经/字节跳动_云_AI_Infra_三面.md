---
title: '字节跳动 云 AI Infra 三面'
date: 2026-04-16 11:43:02
categories:
  - [求职面试, AI Infra, 大厂真题]
tags: [AIInfra, 推理优化, 高性能计算, 大厂面经, 面经, 牛客]
---

# 字节跳动 云 AI Infra 三面

<!-- more -->

> 原文链接: https://www.nowcoder.com/discuss/872091889137156096
> 来源: 牛客网

---

🎯【字节跳动 2025 年 11 月三面真题】大模型推理服务的弹性伸缩与成本优化系统

💼 岗位：AI 平台研发/云原生工程师

🌟 难度：⭐⭐⭐⭐⭐

📌 面试题

“设计一个面向大模型推理服务（如 ERNIE 4.0 的 API 服务）的弹性伸缩与成本优化系统。要求：

1️⃣ 能根据实时 QPS、GPU 利用率、请求延迟等指标，自动扩缩容服务实例（Pod）；

2️⃣ 考虑混合部署，将高优请求路由到 GPU 实例，低优或简单请求路由到优化后的 CPU 实例以节省成本；

3️⃣ 设计一套策略，在业务低峰期自动缩容至最小集群，并在高峰期到来前预热扩容。给出架构设计、核心调度算法，并分析如何平衡性能与成本。”

💡 解析

这题可是资源治理与架构设计领域的典型难题😣，精准戳中了 AI 时代企业面临的核心痛点——算力成本高昂💸。它对候选人的要求可不低，不仅要精通微服务和 K8s，还得具备产品经理的成本意识以及架构师的全局视野👀。

🧠 设计思路拆解
📊 监控与决策闭环
📈 数据采集：借助 Prometheus 全面监控所有模型服务实例的 QPS、P99 延迟、GPU 利用率、显存使用率等关键指标📋。
🧠 决策中心：打造一个独立的 Auto - Scaler 服务，每隔 30 秒（周期可灵活调整🕙）拉取聚合指标，依据预设策略（例如平均 GPU 利用率超过 70%且持续 2 分钟就进行扩容）做出精准的伸缩决策📜。
🛠️ 执行层：通过调用 Kubernetes API，巧妙调整对应 Deployment 的副本数，或者更高级地调整 HPA（Horizontal Pod Autoscaler）的目标值，实现服务的灵活伸缩📈。
🚦 混合调度与路由
🧩 服务部署：同时部署两类服务，gpu - service 以高性能著称但成本较高💎，cpu - optimized - service 则成本较低，可能采用量化模型📉。
🌐 智能路由：在 API 网关或服务网格中实现智能路由功能📡。根据请求 Header 中的优先级标签，或者借助模型预测的请求复杂度，将流量精准分发到不同的后端服务📤。
⏰ 预测性伸缩
📊 流量预测：基于过去 7 天同一时段的 QPS 等历史流量数据📅，运用时间序列预测算法（如 Facebook 的 Prophet）预测未来流量走势📈。
🚀 提前扩容：在预测的流量高峰到来前提前 5 分钟（时间可按需调整⏱️）进行扩容，同时进行模型预热（将权重加载到显存），有效避免冷启动导致的性能毛刺📉。
🎯 应用业务场景

这可是火山引擎 AI 中台和字节内部 AI 平台必须攻克的难题😎。以“抖音特效”为例，白天和夜晚的用户请求量差异巨大🌓。通过弹性伸缩，夜间可以释放大量 GPU 资源用于模型训练📚，白天再快速扩容服务实例，能够节省 30%以上的云资源成本💰。混合调度则确保 VIP 用户或复杂特效请求始终能得到 GPU 的有力保障💪。

📚 核心考点聚焦
Kubernetes 弹性伸缩原理与实践（HPA, VPA, Cluster Autoscaler）📖
云原生监控体系（Prometheus, Metrics Server）📊
流量调度与服务治理（服务网格、网关策略）🌐
成本优化模型与容量规划💰
时间序列预测的基本概念📈
💡 实践避坑指南
📈 避免抖动：伸缩策略一定要设置冷却时间和滞回区间（例如扩容阈值设为 70%，缩容阈值设为 30%），防止指标在阈值附近波动时实例数频繁震荡📉。
🎯 优雅上下线：缩容前，必须通过就绪探针和服务注册中心确保待删除 Pod 已从流量池中摘除🚫，并等待其处理完现有请求，避免流量丢失📤。
💰 成本核算：系统要能够输出详细的资源使用报告和成本分摊情况，这可是向业务方证明自身价值的关键🔑。
🚨 趋势押题预测
📌 预测名称

跨云跨区域智能算力调度与负载均衡

📝 押题题目

“设计一个跨云厂商（如火山引擎、AWS）和跨区域的智能算力调度系统。核心目标是：

1️⃣ 根据各区域/云厂商的实时资源价格、网络延迟、模型副本分布，动态为 AI 推理请求选择最优的服务端点；

2️⃣ 在某个区域故障时，实现秒级流量切换与灾难恢复。阐述整体架构、调度决策算法，以及如何保证数据在跨云传输时的安全与低延迟。”

📊 押题依据
📈 频次统计：在高级架构师面试中，“多云/混合云架构”和“成本与容灾”是紧密关联的顶级考点，每年出现高达 15 次，是体现技术视野广度的标志性题目🏆。
🌍 新趋势需求：字节业务走向全球化，必须避免单一云厂商绑定，同时充分利用不同区域的廉价算力时段🕙。“降本增效” 和 “异地多活” 是 2026 年所有大厂技术战略的重中之重💯。
📚 信息来源：参考业界多云管理平台实践，以及大型互联网公司（包括字节）在财报和分享中频繁提及的“基础设施优化”方向🧭。
💡 押题逻辑理由

上一题解决的是单集群内的弹性伸缩问题📊。而更宏观、更复杂的挑战在于跨集群、跨云、跨地域的资源调度🌐。这涉及到网络、财务、安全、合规等多个维度，是真正意义上的 CTO 级别的架构思考🤔。能清晰阐述此类方案的候选人，展现的是领导一个技术方向所需的战略思维和系统整合能力💪。

📚 核心考点聚焦

多云架构、智能 DNS/GSLB、成本优化算法、异地多活容灾、零信任安全网络🔒

💼 适配岗位

云原生架构师、基础架构负责人、SRE 专家👨‍💻

📈 押中概率

65%（战略级架构题，用于选拔技术负责人或资深专家🧑‍💼）

	
// 【代码示例】智能弹性伸缩决策器核心片段
@Service
public class IntelligentAutoScaler {
    @Autowired
    private KubernetesApiService k8sService;
    @Autowired
    private MetricService metricService;
    @Autowired
    private CostCalculator costCalculator;
 
    @Scheduled(fixedDelay = 30000) // 每30秒执行一次决策
    public void scalingDecision() {
        String deploymentName = "ernie-api-gpu";
        ScalingDecision decision = new ScalingDecision();
 
        // 1. 获取实时指标
        double currentGpuUtil = metricService.getAverageGpuUtil(deploymentName);
        long currentQps = metricService.getCurrentQps(deploymentName);
        double predictedQps = metricService.getPredictedQps(); // 来自预测模型
 
        // 2. 基于规则的决策 (示例)
        if (currentGpuUtil > 75 || predictedQps > currentQps * 1.3) {
            decision.setAction(ScalingAction.SCALE_OUT);
            decision.setReason("高负载或预测流量增长");
        } else if (currentGpuUtil < 25 && currentQps < 100) {
            // 3. 成本感知决策：判断缩容是否真的省钱（考虑预留实例）
            if (costCalculator.willSaveCostByScalingIn(deploymentName)) {
                decision.setAction(ScalingAction.SCALE_IN);
                decision.setReason("低负载且缩容符合成本效益");
            }
        } else {
            decision.setAction(ScalingAction.HOLD);
        }
 
        // 4. 执行决策 (带冷却检查)
        if (decision.getAction() != ScalingAction.HOLD && !isInCooldown(deploymentName)) {
            executeScaling(deploymentName, decision);
            recordCooldown(deploymentName);
        }
    }
 
    private void executeScaling(String deployment, ScalingDecision decision) {
        int currentReplicas = k8sService.getReplicas(deployment);
        int newReplicas = currentReplicas;
        switch (decision.getAction()) {
            case SCALE_OUT:
                newReplicas = Math.min(currentReplicas + 2, 20); // 最多扩到20个实例
                break;
            case SCALE_IN:
                newReplicas = Math.max(currentReplicas - 1, 2); // 最少保留2个实例保高可用
                break;
        }
        k8sService.scaleDeployment(deployment, newReplicas);
        // 发送事件通知
        eventBus.post(new ScalingEvent(deployment, newReplicas, decision.getReason()));
    }
}


~~~



字节跳动真题详解+代码+押题集
