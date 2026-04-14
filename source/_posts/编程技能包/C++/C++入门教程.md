---
title: C++入门教程
date: 2026-02-09 12:15:00
categories:
  - [编程技能包, C++基础]
tags: [C++, 编程语言, 入门教程, 面向对象]
---

C++ 是一门功能强大的通用编程语言，广泛应用于系统软件、游戏开发、嵌入式系统和高性能计算等领域。本文将带你快速入门 C++ 编程。

<!-- more -->

## ❓ 什么是 C++？

C++ 由 Bjarne Stroustrup 于 1979 年在贝尔实验室开发，是 C 语言的扩展。如果把编程语言比作工具，C 就是一把精密的手术刀，而 C++ 则是在这把手术刀的基础上加了瑞士军刀的多功能性。它支持：

- **面向过程编程**：继承自 C 语言
- **面向对象编程**：类、继承、多态、封装
- **泛型编程**：模板（Templates）
- **函数式编程**：Lambda 表达式（C++11）

## ⚙️ 环境配置

### 1. 安装编译器

**Linux (Ubuntu/Debian)**：
```bash
sudo apt update
sudo apt install g++ build-essential
```

**macOS**：
```bash
xcode-select --install
```

**Windows**：
- 安装 [MinGW-w64](https://www.mingw-w64.org/) 或
- 安装 [Visual Studio](https://visualstudio.microsoft.com/)

### 2. 验证安装

```bash
g++ --version
```

## 💻 第一个 C++ 程序

创建文件 `hello.cpp`：

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
```

编译并运行：

```bash
g++ hello.cpp -o hello
./hello
```

## 💻 基本语法

### 变量与数据类型

```cpp
#include <iostream>
#include <string>

int main() {
    // 基本数据类型
    int age = 25;              // 整数
    double price = 99.99;      // 双精度浮点数
    float rate = 3.14f;        // 单精度浮点数
    char grade = 'A';          // 字符
    bool isActive = true;      // 布尔值
    
    // 字符串（需要 #include <string>）
    std::string name = "张三";
    
    std::cout << "姓名: " << name << std::endl;
    std::cout << "年龄: " << age << std::endl;
    
    return 0;
}
```

### 控制流语句

**条件语句**：

```cpp
int score = 85;

if (score >= 90) {
    std::cout << "优秀" << std::endl;
} else if (score >= 60) {
    std::cout << "及格" << std::endl;
} else {
    std::cout << "不及格" << std::endl;
}
```

**循环语句**：

```cpp
// for 循环
for (int i = 0; i < 5; i++) {
    std::cout << i << " ";
}

// while 循环
int j = 0;
while (j < 5) {
    std::cout << j << " ";
    j++;
}

// 范围 for 循环 (C++11)
int arr[] = {1, 2, 3, 4, 5};
for (int num : arr) {
    std::cout << num << " ";
}
```

### 函数

```cpp
#include <iostream>

// 函数声明
int add(int a, int b);
void greet(std::string name);

int main() {
    int result = add(3, 5);
    std::cout << "3 + 5 = " << result << std::endl;
    
    greet("小明");
    return 0;
}

// 函数定义
int add(int a, int b) {
    return a + b;
}

void greet(std::string name) {
    std::cout << "你好, " << name << "!" << std::endl;
}
```

## 💻 面向对象编程

### 类与对象

```cpp
#include <iostream>
#include <string>

class Person {
private:
    std::string name;
    int age;

public:
    // 构造函数
    Person(std::string n, int a) : name(n), age(a) {}
    
    // 成员函数
    void introduce() {
        std::cout << "我是 " << name << "，今年 " << age << " 岁。" << std::endl;
    }
    
    // Getter
    std::string getName() { return name; }
    int getAge() { return age; }
    
    // Setter
    void setAge(int a) { age = a; }
};

int main() {
    Person p1("张三", 25);
    p1.introduce();
    
    p1.setAge(26);
    std::cout << "明年 " << p1.getName() << " 就 " << p1.getAge() << " 岁了。" << std::endl;
    
    return 0;
}
```

### 继承

```cpp
#include <iostream>

// 基类
class Animal {
protected:
    std::string name;
    
public:
    Animal(std::string n) : name(n) {}
    
    virtual void speak() {
        std::cout << name << " 发出声音" << std::endl;
    }
};

// 派生类
class Dog : public Animal {
public:
    Dog(std::string n) : Animal(n) {}
    
    void speak() override {
        std::cout << name << " 汪汪叫" << std::endl;
    }
};

class Cat : public Animal {
public:
    Cat(std::string n) : Animal(n) {}
    
    void speak() override {
        std::cout << name << " 喵喵叫" << std::endl;
    }
};

int main() {
    Dog dog("旺财");
    Cat cat("咪咪");
    
    dog.speak();  // 旺财 汪汪叫
    cat.speak();  // 咪咪 喵喵叫
    
    return 0;
}
```

## ⚙️ 现代 C++ 特性 (C++11/14/17/20)

### 智能指针

**智能指针**是自动管理内存的指针，用完会自动释放——就像有自动回收功能的垃圾桶，你把东西扔进去就不用管了，它会在合适的时候自己清空。这背后体现了 C++ 中一个重要的设计思想：**RAII（资源获取即初始化）**——对象创建时自动获取资源，销毁时自动释放，就像进门自动开灯、出门自动关灯，不需要你手动操心。

```cpp
#include <iostream>
#include <memory>

class MyClass {
public:
    MyClass() { std::cout << "构造" << std::endl; }
    ~MyClass() { std::cout << "析构" << std::endl; }
    void hello() { std::cout << "Hello!" << std::endl; }
};

int main() {
    // unique_ptr - 独占所有权
    std::unique_ptr<MyClass> ptr1 = std::make_unique<MyClass>();
    ptr1->hello();
    
    // shared_ptr - 共享所有权
    std::shared_ptr<MyClass> ptr2 = std::make_shared<MyClass>();
    std::shared_ptr<MyClass> ptr3 = ptr2;  // 引用计数 +1
    
    return 0;
}  // 自动释放内存
```

### Lambda 表达式

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> nums = {3, 1, 4, 1, 5, 9, 2, 6};
    
    // Lambda 排序
    std::sort(nums.begin(), nums.end(), [](int a, int b) {
        return a > b;  // 降序
    });
    
    // Lambda 遍历
    std::for_each(nums.begin(), nums.end(), [](int n) {
        std::cout << n << " ";
    });
    
    return 0;
}
```

### auto 关键字

```cpp
auto x = 10;           // int
auto y = 3.14;         // double
auto str = "hello";    // const char*

std::vector<int> vec = {1, 2, 3};
for (auto& num : vec) {
    num *= 2;
}
```

## 📚 常用标准库

C++ 的标准库中最核心的部分叫 **STL（Standard Template Library，标准模板库）**——你可以把它理解为 C++ 的"标准工具箱"，里面有各种现成的数据结构和算法，拿来直接用，不用自己从零造轮子。

| 头文件 | 功能 |
|--------|------|
| `<iostream>` | 输入输出流 |
| `<string>` | 字符串处理 |
| `<vector>` | 动态数组 |
| `<map>` | 键值对容器 |
| `<algorithm>` | 常用算法 |
| `<memory>` | 智能指针 |
| `<thread>` | 多线程 |
| `<fstream>` | 文件操作 |

## 🔧 编译选项建议

```bash
# 启用 C++17 标准，开启所有警告
g++ -std=c++17 -Wall -Wextra -o program source.cpp

# 开启优化
g++ -std=c++17 -O2 -o program source.cpp
```

## 🎯 自我检验清单

学完本文后，你可以用以下清单检验自己的掌握程度：

- [ ] 能独立配置 C++ 编译环境，使用 `g++` 编译并运行一个 C++ 程序
- [ ] 能熟练使用基本数据类型（`int`、`double`、`string`、`bool` 等）声明变量并完成输入输出
- [ ] 能使用 `if/else`、`for`、`while` 等控制流语句编写逻辑
- [ ] 能定义和调用函数，理解函数声明与定义的区别
- [ ] 能定义类，使用构造函数、成员函数、访问控制（`public`/`private`）封装数据
- [ ] 能使用继承和 `virtual`/`override` 实现多态
- [ ] 能使用 `std::unique_ptr` 和 `std::shared_ptr` 管理动态内存，理解 RAII 思想
- [ ] 能使用 STL 容器（`vector`、`map`）和算法（`sort`、`for_each`），配合 Lambda 表达式完成常见任务

## 📚 参考资料

- [cppreference.com](https://en.cppreference.com/) - C++ 权威参考手册，涵盖标准库和语言特性的完整文档
- [LearnCpp](https://www.learncpp.com/) - 免费在线教程，从零开始系统学习 C++，内容详尽且持续更新
- [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) - 由 Bjarne Stroustrup 和 Herb Sutter 维护的 C++ 编程最佳实践指南
- [Compiler Explorer (Godbolt)](https://godbolt.org/) - 在线编译器，可以实时查看 C++ 代码生成的汇编指令
- 《C++ Primer》（第 5 版）- Stanley B. Lippman 等著，公认的 C++ 经典入门书籍
- 《Effective C++》- Scott Meyers 著，55 条改善程序与设计的具体做法
- 《Effective Modern C++》- Scott Meyers 著，聚焦 C++11/14 的最佳实践

## 📝 总结

C++ 是一门功能强大但学习曲线较陡的语言。建议学习路线：

1. **基础语法** → 变量、控制流、函数
2. **面向对象** → 类、继承、多态
3. **内存管理** → 指针、引用、智能指针
   - **指针**：存储内存地址的变量——就像写着"某某在 3 楼 302 房间"的纸条，拿着纸条就能找到那个人
   - **引用**：给变量起的"别名"——就像一个人的小名和大名，叫哪个都是同一个人
4. **标准库** → STL 容器和算法
5. **现代特性** → C++11/14/17/20 新特性

掌握这些基础后，你就可以开始编写实际的 C++ 项目了！
