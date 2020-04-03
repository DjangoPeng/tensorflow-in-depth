<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [深入理解 TensorFlow 架构设计与实现原理](#%E6%B7%B1%E5%85%A5%E7%90%86%E8%A7%A3-tensorflow-%E6%9E%B6%E6%9E%84%E8%AE%BE%E8%AE%A1%E4%B8%8E%E5%AE%9E%E7%8E%B0%E5%8E%9F%E7%90%86)
  - [推荐语(节选)](#%E6%8E%A8%E8%8D%90%E8%AF%AD%E8%8A%82%E9%80%89)
  - [样例代码（已开源）](#%E6%A0%B7%E4%BE%8B%E4%BB%A3%E7%A0%81%E5%B7%B2%E5%BC%80%E6%BA%90)
  - [本书目录](#%E6%9C%AC%E4%B9%A6%E7%9B%AE%E5%BD%95)
    - [前言（开源）](#%E5%89%8D%E8%A8%80%E5%BC%80%E6%BA%90)
    - [第一部分 基础篇](#%E7%AC%AC%E4%B8%80%E9%83%A8%E5%88%86-%E5%9F%BA%E7%A1%80%E7%AF%87)
      - [第1章 TensorFlow系统概述（开源）](#%E7%AC%AC1%E7%AB%A0-tensorflow%E7%B3%BB%E7%BB%9F%E6%A6%82%E8%BF%B0%E5%BC%80%E6%BA%90)
      - [第2章 TensorFlow环境准备](#%E7%AC%AC2%E7%AB%A0-tensorflow%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)
      - [第3章 TensorFlow基础概念](#%E7%AC%AC3%E7%AB%A0-tensorflow%E5%9F%BA%E7%A1%80%E6%A6%82%E5%BF%B5)
    - [第二部分 关键模块篇](#%E7%AC%AC%E4%BA%8C%E9%83%A8%E5%88%86-%E5%85%B3%E9%94%AE%E6%A8%A1%E5%9D%97%E7%AF%87)
      - [第4章 TensorFlow数据处理方法](#%E7%AC%AC4%E7%AB%A0-tensorflow%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86%E6%96%B9%E6%B3%95)
      - [第5章 TensorFlow编程框架](#%E7%AC%AC5%E7%AB%A0-tensorflow%E7%BC%96%E7%A8%8B%E6%A1%86%E6%9E%B6)
      - [第6章 TensorBoard可视化工具](#%E7%AC%AC6%E7%AB%A0-tensorboard%E5%8F%AF%E8%A7%86%E5%8C%96%E5%B7%A5%E5%85%B7)
      - [第7章 TensorFlow模型托管工具](#%E7%AC%AC7%E7%AB%A0-tensorflow%E6%A8%A1%E5%9E%8B%E6%89%98%E7%AE%A1%E5%B7%A5%E5%85%B7)
    - [第三部分 算法模型篇](#%E7%AC%AC%E4%B8%89%E9%83%A8%E5%88%86-%E7%AE%97%E6%B3%95%E6%A8%A1%E5%9E%8B%E7%AF%87)
      - [第8章 深度学习概述](#%E7%AC%AC8%E7%AB%A0-%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A6%82%E8%BF%B0)
      - [第9章 卷积神经网络](#%E7%AC%AC9%E7%AB%A0-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
      - [第10章 生成对抗网络](#%E7%AC%AC10%E7%AB%A0-%E7%94%9F%E6%88%90%E5%AF%B9%E6%8A%97%E7%BD%91%E7%BB%9C)
      - [第11章 循环神经网络](#%E7%AC%AC11%E7%AB%A0-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
    - [第四部分 核心揭秘篇](#%E7%AC%AC%E5%9B%9B%E9%83%A8%E5%88%86-%E6%A0%B8%E5%BF%83%E6%8F%AD%E7%A7%98%E7%AF%87)
      - [第12章 TensorFlow运行时核心设计与实现](#%E7%AC%AC12%E7%AB%A0-tensorflow%E8%BF%90%E8%A1%8C%E6%97%B6%E6%A0%B8%E5%BF%83%E8%AE%BE%E8%AE%A1%E4%B8%8E%E5%AE%9E%E7%8E%B0)
      - [第13章 通信原理与实现](#%E7%AC%AC13%E7%AB%A0-%E9%80%9A%E4%BF%A1%E5%8E%9F%E7%90%86%E4%B8%8E%E5%AE%9E%E7%8E%B0)
      - [第14章 数据流图计算原理与实现](#%E7%AC%AC14%E7%AB%A0-%E6%95%B0%E6%8D%AE%E6%B5%81%E5%9B%BE%E8%AE%A1%E7%AE%97%E5%8E%9F%E7%90%86%E4%B8%8E%E5%AE%9E%E7%8E%B0)
    - [第五部分 生态发展篇](#%E7%AC%AC%E4%BA%94%E9%83%A8%E5%88%86-%E7%94%9F%E6%80%81%E5%8F%91%E5%B1%95%E7%AF%87)
      - [第15章 TensorFlow生态环境](#%E7%AC%AC15%E7%AB%A0-tensorflow%E7%94%9F%E6%80%81%E7%8E%AF%E5%A2%83)
  - [参考链接](#%E5%8F%82%E8%80%83%E9%93%BE%E6%8E%A5)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# 深入理解 TensorFlow 架构设计与实现原理

此项目托管了《深入理解 TensorFlow 架构设计与实现原理》一书的样章与部分示例代码。

## [推荐语(节选)](recommendations.md)

*"很高兴能够在这个时候看到一本讲授如何使用TensorFlow的专业书籍。作者是深谙计算机系统之道的一线工程师，带给读者的是产生自实战经验基础上的理解。非常难得的是，本书除了讲解如何使用TensorFlow还加入了对系统设计原理方面的剖析，有助于读者做针对性的应用和系统优化。相信本书对从事深度学习方面研究和开发的读者定会有所裨益。"*

—— 查礼 中国科学院计算技术研究所 副研究员 中国大数据技术大会（BDTC） 发起人

*“TensorFlow还是一个较新的技术，但是发展极为迅猛，在这时候出现一本深入浅出讲解TensorFlow理论与应用的书籍，对于广大希望学习和应用大数据深度学习技术的读者而言，诚“如大旱之望云霓”。本书理论与实践并重，理论上讲清楚了一些本质的东西，并加入了作者对系统设计原理方面的深刻理解，并通过实际案例，引导读者掌握针对性的系统优化的技能。*

*本书第一作者是我的学生，12年入学时进入了浙大学习计算机专业的尖子班“求是科学班”，我担任了他们这个班的班主任。他不仅品学兼优，而且作为班上的团支书，帮我这个不太称职的班主任做了很多班级工作。在我心目中，他依然是入学时的青涩模样，转眼间却已成为开源软件届的技术翘楚，作为老师，最欣慰的莫过于此了吧。是为序。”*

—— 陈刚 教育部“长江学者”特聘教授 浙江大学计算机学院院长   

## [样例代码（已开源）](code/)

**说明：代码文件格式为 "章节_代码名称"，如3.6节线性回归最佳实践代码文件名为 "3.6\_best\_practice.py"。**

## [本书目录](contents.md)

### [前言（开源）](preface.md)

### 第一部分 基础篇

#### [第1章 TensorFlow系统概述（开源）](text/1_overview/1.0_overview.md)

#### 第2章 TensorFlow环境准备

#### 第3章 TensorFlow基础概念

### 第二部分 关键模块篇

#### 第4章 TensorFlow数据处理方法

#### 第5章 TensorFlow编程框架

#### 第6章 TensorBoard可视化工具

#### 第7章 TensorFlow模型托管工具

### 第三部分 算法模型篇

#### 第8章 深度学习概述

#### 第9章 卷积神经网络

#### 第10章 生成对抗网络

#### 第11章 循环神经网络

### 第四部分 核心揭秘篇

#### 第12章 TensorFlow运行时核心设计与实现

#### 第13章 通信原理与实现

#### 第14章 数据流图计算原理与实现

### 第五部分 生态发展篇

#### 第15章 TensorFlow生态环境


## 参考链接

- [人民邮电出版社官方介绍](http://www.ptpress.com.cn/shopping/buy?bookId=d87d343a-66f0-4430-b48d-4d03273f8258)
- [京东商城链接](http://item.jd.com/12349620.html)

