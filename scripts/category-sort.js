'use strict';

/**
 * 自定义分类排序脚本
 * 支持两级树状分类结构展示
 */

// 一级分类的展示顺序
const topLevelOrder = ['AI Infra', '路飞玩AI', '编程技能包'];

// 二级分类的展示顺序（按一级分类分组，顺序与 categories.md 一致）
const subCategoryOrder = {
  'AI Infra': ['学习路线', '前置知识', 'CUDA编程与算子优化', '分布式训练', '推理与部署', '性能分析'],
  '路飞玩AI': ['AI编程', 'Agent开发'],
  '编程技能包': ['Python', 'C++基础', 'Web开发']
};

// 二级分类排序函数
function sortSubCategories(children, parentName) {
  const order = subCategoryOrder[parentName] || [];
  children.sort((a, b) => {
    const idxA = order.indexOf(a.name);
    const idxB = order.indexOf(b.name);
    if (idxA !== -1 && idxB !== -1) return idxA - idxB;
    if (idxA !== -1) return -1;
    if (idxB !== -1) return 1;
    return a.name.localeCompare(b.name, 'zh-CN');
  });
}

// 一级分类的图标映射
const categoryIcons = {
  'AI Infra': 'fa-server',
  '路飞玩AI': 'fa-robot',
  '编程技能包': 'fa-code'
};

// 注册树状分类数据 helper（首页 README 卡片使用）
// 返回 [{ name, icon, children: [{ category, posts }] }]
hexo.extend.helper.register('sorted_categories_tree', function() {
  const categories = this.site.categories.toArray();

  // 分离一级和二级分类
  const topLevel = categories.filter(c => !c.parent);
  const childrenMap = {};

  categories.forEach(cat => {
    if (cat.parent) {
      if (!childrenMap[cat.parent]) {
        childrenMap[cat.parent] = [];
      }
      childrenMap[cat.parent].push(cat);
    }
  });

  // 按预定义顺序排列一级分类
  topLevel.sort((a, b) => {
    const idxA = topLevelOrder.indexOf(a.name);
    const idxB = topLevelOrder.indexOf(b.name);
    if (idxA !== -1 && idxB !== -1) return idxA - idxB;
    if (idxA !== -1) return -1;
    if (idxB !== -1) return 1;
    return a.name.localeCompare(b.name, 'zh-CN');
  });

  return topLevel.map(parent => {
    const children = childrenMap[parent._id] || [];
    sortSubCategories(children, parent.name);
    return {
      name: parent.name,
      icon: categoryIcons[parent.name] || 'fa-folder-open',
      totalPosts: children.length > 0
        ? children.reduce((sum, c) => sum + c.length, 0)
        : parent.length,
      children: children
    };
  });
});

// 一级分类的介绍文案
const categoryDescriptions = {
  'AI Infra': '涵盖计算机底层基础、大模型训练部署和 CUDA GPU 编程等基础设施技术。',
  '路飞玩AI': '探索 AI 编程工具的最佳实践与 Agent 智能体的设计开发。',
  '编程技能包': '实用编程技能：Web 开发、Python 生态、工具链与工程实践。'
};

// 一级分类对应的着陆页路径
const categoryLandingPaths = {
  'AI Infra': '/ai-infra/',
  '路飞玩AI': '/play-ai/',
  '编程技能包': '/coding-skills/'
};

// 获取一级分类第一篇文章的 URL（导航栏直链使用）
hexo.extend.helper.register('get_category_first_post_url', function(categoryName) {
  const categories = this.site.categories.toArray();
  const parent = categories.find(c => c.name === categoryName && !c.parent);
  if (!parent) return categoryLandingPaths[categoryName] || '/';

  const children = categories.filter(c => c.parent === parent._id);
  sortSubCategories(children, parent.name);

  for (const child of children) {
    const posts = child.posts.toArray().sort((a, b) => b.date - a.date);
    if (posts.length > 0) {
      return this.url_for(posts[0].path);
    }
  }
  return categoryLandingPaths[categoryName] || '/';
});

// 根据文章获取其所属一级分类的 sidebar 数据（文章详情页使用）
hexo.extend.helper.register('get_post_category_sidebar', function(post) {
  if (!post.categories || !post.categories.length) return null;

  const allCategories = this.site.categories.toArray();
  const postCats = post.categories.toArray();

  // 找到该文章所属的一级分类（无 parent 的分类）
  let topCat = null;
  for (const cat of postCats) {
    if (!cat.parent) {
      topCat = cat;
      break;
    }
  }
  // 如果没有直接的一级分类，通过子分类的 parent 找到
  if (!topCat) {
    for (const cat of postCats) {
      if (cat.parent) {
        topCat = allCategories.find(c => c._id === cat.parent);
        if (topCat) break;
      }
    }
  }
  if (!topCat) return null;

  // 复用 get_category_landing 的逻辑
  const children = allCategories.filter(c => c.parent === topCat._id);
  sortSubCategories(children, topCat.name);

  const childrenData = children.map(c => ({
    name: c.name,
    path: c.path,
    posts: c.posts.toArray().sort((a, b) => b.date - a.date)
  }));

  // 取第一个子分类的第一篇文章作为分类入口
  let landingPath = categoryLandingPaths[topCat.name] || '/categories/';
  for (const child of childrenData) {
    if (child.posts.length > 0) {
      landingPath = this.url_for(child.posts[0].path);
      break;
    }
  }

  return {
    name: topCat.name,
    icon: categoryIcons[topCat.name] || 'fa-folder-open',
    landingPath: landingPath,
    children: childrenData,
    totalPosts: children.reduce((sum, c) => sum + c.length, 0)
  };
});

// 获取一级分类下的所有文章（含子分类），按时间倒序
// 返回 { name, icon, description, children: [{ name, posts }], allPosts }
hexo.extend.helper.register('get_category_landing', function(categoryName) {
  const categories = this.site.categories.toArray();

  // 找到对应的一级分类
  const parent = categories.find(c => c.name === categoryName && !c.parent);
  if (!parent) return null;

  // 找到所有子分类
  const children = categories.filter(c => c.parent === parent._id);
  sortSubCategories(children, parent.name);

  // 收集所有文章并去重
  const postMap = new Map();
  children.forEach(child => {
    child.posts.forEach(post => {
      if (!postMap.has(post._id)) {
        postMap.set(post._id, { post, subCategory: child.name });
      }
    });
  });
  // 也加入直接在父分类下的文章
  parent.posts.forEach(post => {
    if (!postMap.has(post._id)) {
      postMap.set(post._id, { post, subCategory: parent.name });
    }
  });

  // 按时间倒序
  const allPosts = Array.from(postMap.values());
  allPosts.sort((a, b) => b.post.date - a.post.date);

  return {
    name: parent.name,
    icon: categoryIcons[parent.name] || 'fa-folder-open',
    description: categoryDescriptions[parent.name] || '',
    children: children.map(c => ({
      name: c.name,
      path: c.path,
      posts: c.posts.toArray().sort((a, b) => b.date - a.date)
    })),
    allPosts: allPosts
  };
});

// 注册树状分类列表 helper（分类页使用）
hexo.extend.helper.register('list_categories_sorted', function() {
  const categories = this.site.categories.toArray();

  if (!categories || categories.length === 0) {
    return '';
  }

  // 分离一级分类（无 parent）和二级分类（有 parent）
  const topLevel = categories.filter(c => !c.parent);
  const childrenMap = {};

  categories.forEach(cat => {
    if (cat.parent) {
      if (!childrenMap[cat.parent]) {
        childrenMap[cat.parent] = [];
      }
      childrenMap[cat.parent].push(cat);
    }
  });

  // 按预定义顺序排列一级分类
  topLevel.sort((a, b) => {
    const idxA = topLevelOrder.indexOf(a.name);
    const idxB = topLevelOrder.indexOf(b.name);
    if (idxA !== -1 && idxB !== -1) return idxA - idxB;
    if (idxA !== -1) return -1;
    if (idxB !== -1) return 1;
    return a.name.localeCompare(b.name, 'zh-CN');
  });

  // 生成树状 HTML
  let result = '<div class="category-tree">';

  topLevel.forEach(parent => {
    const children = childrenMap[parent._id] || [];
    const totalPosts = children.length > 0
      ? children.reduce((sum, c) => sum + c.length, 0)
      : parent.length;
    const icon = categoryIcons[parent.name] || 'fa-folder-open';

    // 按 categories.md 预定义顺序排列子分类
    sortSubCategories(children, parent.name);

    result += '<div class="category-tree-group">';
    result += '<div class="category-tree-parent">';
    result += `<i class="fa ${icon} category-tree-icon"></i>`;
    result += `<span class="category-tree-name">${parent.name}</span>`;
    result += `<span class="category-tree-count">${totalPosts} 篇</span>`;
    result += '</div>';

    if (children.length > 0) {
      result += '<ul class="category-tree-children">';
      children.forEach((child, idx) => {
        const isLast = idx === children.length - 1;
        const url = this.url_for(child.path);
        result += `<li class="category-tree-child${isLast ? ' last' : ''}">`;
        result += `<span class="category-tree-branch">${isLast ? '└──' : '├──'}</span>`;
        result += `<a href="${url}" class="category-tree-link">${child.name}</a>`;
        result += `<span class="category-tree-child-count">${child.length}</span>`;
        result += '</li>';
      });
      result += '</ul>';
    }

    result += '</div>';
  });

  result += '</div>';
  return result;
});
