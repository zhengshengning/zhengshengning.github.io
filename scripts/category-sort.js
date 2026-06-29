'use strict';

/**
 * 自定义分类排序脚本
 * 支持两级树状分类结构展示
 */

// 一级分类的展示顺序
const topLevelOrder = ['AI Infra', '求职面试', '路飞玩AI', '编程技能包'];

// 二级分类的展示顺序（按一级分类分组，顺序与 categories.md 一致）
const subCategoryOrder = {
  'AI Infra': ['学习指南', '前置知识', 'CUDA编程与算子优化', '分布式训练', '推理与部署', '性能分析'],
  '求职面试': ['大厂面经', '知名科技公司面经', '车企/自驾面经', 'AI 创业公司面经', '芯片/研究院面经', '其他公司面经', '综合面经'],
  '路飞玩AI': ['AI编程', 'Agent开发'],
  '编程技能包': ['Python', 'C++基础', 'Web开发']
};

// 三级分类的展示顺序（按二级分类分组）
const level3CategoryOrder = {
  '前置知识': ['编程基础', '深度学习基础', 'Transformer', 'Pytorch']
};

// 文章排序比较函数：优先按 order 数值升序，其次按标题（含数字感知）
// - 同时有 order：按 order 升序
// - 只有一方有 order：有 order 的排前面
// - 都没有 order：按标题做数字感知比较（"第2章" < "第11章"）
function comparePosts(a, b) {
  const hasA = typeof a.order === 'number' && !isNaN(a.order);
  const hasB = typeof b.order === 'number' && !isNaN(b.order);
  if (hasA && hasB) {
    if (a.order !== b.order) return a.order - b.order;
    return a.title.localeCompare(b.title, 'zh-CN', { numeric: true });
  }
  if (hasA) return -1;
  if (hasB) return 1;
  return a.title.localeCompare(b.title, 'zh-CN', { numeric: true });
}

// 分类名兜底比较：数字感知，保证 "第2章" < "第11章"、"1-基础" < "3-高阶"
function compareCategoryName(a, b) {
  return a.name.localeCompare(b.name, 'zh-CN', { numeric: true });
}

// 二级分类排序函数
function sortSubCategories(children, parentName) {
  const order = subCategoryOrder[parentName] || [];
  children.sort((a, b) => {
    const idxA = order.indexOf(a.name);
    const idxB = order.indexOf(b.name);
    if (idxA !== -1 && idxB !== -1) return idxA - idxB;
    if (idxA !== -1) return -1;
    if (idxB !== -1) return 1;
    return compareCategoryName(a, b);
  });
}

// 三级分类排序函数
function sortLevel3Categories(children, parentName) {
  const order = level3CategoryOrder[parentName] || [];
  children.sort((a, b) => {
    const idxA = order.indexOf(a.name);
    const idxB = order.indexOf(b.name);
    if (idxA !== -1 && idxB !== -1) return idxA - idxB;
    if (idxA !== -1) return -1;
    if (idxB !== -1) return 1;
    return compareCategoryName(a, b);
  });
}

// 构建二级分类数据（支持三级子分类）
// 返回 { hasSubGroups, subGroups?, directPosts?, posts? }
function buildLevel2Data(level2Cat, allCategories) {
  const level3Children = allCategories.filter(c => c.parent === level2Cat._id);
  sortLevel3Categories(level3Children, level2Cat.name);

  if (level3Children.length > 0) {
    // 有三级子分类：分离直接文章和子分组文章
    const level3PostIds = new Set();
    const subGroups = level3Children.map(l3 => {
      const posts = l3.posts.toArray().sort(comparePosts);
      posts.forEach(p => level3PostIds.add(p._id));
      return { name: l3.name, posts };
    });
    // 直接挂在二级分类下、不属于任何三级的文章
    const directPosts = level2Cat.posts.toArray()
      .filter(p => !level3PostIds.has(p._id))
      .sort(comparePosts);
    const totalPosts = directPosts.length + subGroups.reduce((sum, sg) => sum + sg.posts.length, 0);
    return { hasSubGroups: true, name: level2Cat.name, path: level2Cat.path, subGroups, directPosts, totalPosts };
  } else {
    // 无三级子分类：与现有结构兼容
    const posts = level2Cat.posts.toArray().sort(comparePosts);
    return { hasSubGroups: false, name: level2Cat.name, path: level2Cat.path, posts, totalPosts: posts.length };
  }
}

// 首页不逐篇展开的分类（只显示分类名和文章数）
const homepageCollapsedCategories = ['求职面试'];

// 一级分类的图标映射
const categoryIcons = {
  'AI Infra': 'fa-server',
  '求职面试': 'fa-briefcase',
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
    // 不在首页逐篇展开的分类，只显示分类名和文章数
    const collapsed = homepageCollapsedCategories.includes(parent.name);
    return {
      name: parent.name,
      icon: categoryIcons[parent.name] || 'fa-folder-open',
      collapsed: collapsed,
      landingPath: categoryLandingPaths[parent.name] || '/categories/',
      totalPosts: children.length > 0
        ? children.reduce((sum, c) => sum + c.length, 0)
        : parent.length,
      children: children.map(child => {
        // 收集该二级分类下的所有文章（含三级子分类），按标题排序
        const level3Children = categories.filter(c => c.parent === child._id);
        const level3PostIds = new Set();
        level3Children.forEach(l3 => {
          l3.posts.forEach(p => level3PostIds.add(p._id));
        });
        // 所有文章 = 直接文章 + 三级子分类文章（去重）
        const allPosts = child.posts.toArray().sort(comparePosts);
        return {
          name: child.name,
          length: child.length,
          path: child.path,
          sortedPosts: allPosts
        };
      })
    };
  });
});

// 一级分类的介绍文案
const categoryDescriptions = {
  'AI Infra': '涵盖计算机底层基础、大模型训练部署和 CUDA GPU 编程等基础设施技术。',
  '求职面试': '汇总 AI Infra 方向的大厂面经、常见面试题和求职经验分享。',
  '路飞玩AI': '探索 AI 编程工具的最佳实践与 Agent 智能体的设计开发。',
  '编程技能包': '实用编程技能：Web 开发、Python 生态、工具链与工程实践。'
};

// 一级分类对应的着陆页路径
const categoryLandingPaths = {
  'AI Infra': '/ai-infra/',
  '求职面试': '/interview/',
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
    const posts = child.posts.toArray().sort(comparePosts);
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
  // 三级文章需要向上遍历到根
  let topCat = null;
  for (const cat of postCats) {
    if (!cat.parent) {
      topCat = cat;
      break;
    }
  }
  // 如果没有直接的一级分类，通过子分类的 parent 向上遍历到根
  if (!topCat) {
    for (const cat of postCats) {
      let current = cat;
      while (current && current.parent) {
        current = allCategories.find(c => c._id === current.parent);
      }
      if (current && !current.parent) {
        topCat = current;
        break;
      }
    }
  }
  if (!topCat) return null;

  // 构建二级分类数据（支持三级）
  const children = allCategories.filter(c => c.parent === topCat._id);
  sortSubCategories(children, topCat.name);

  const childrenData = children.map(c => buildLevel2Data(c, allCategories));

  // 取第一个子分类的第一篇文章作为分类入口
  let landingPath = categoryLandingPaths[topCat.name] || '/categories/';
  for (const child of childrenData) {
    if (child.hasSubGroups) {
      if (child.directPosts.length > 0) {
        landingPath = this.url_for(child.directPosts[0].path);
        break;
      }
      for (const sg of child.subGroups) {
        if (sg.posts.length > 0) {
          landingPath = this.url_for(sg.posts[0].path);
          break;
        }
      }
      if (landingPath !== (categoryLandingPaths[topCat.name] || '/categories/')) break;
    } else {
      if (child.posts.length > 0) {
        landingPath = this.url_for(child.posts[0].path);
        break;
      }
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
// 返回 { name, icon, description, children: [{ name, posts, hasSubGroups?, ... }], allPosts }
hexo.extend.helper.register('get_category_landing', function(categoryName) {
  const categories = this.site.categories.toArray();

  // 找到对应的一级分类
  const parent = categories.find(c => c.name === categoryName && !c.parent);
  if (!parent) return null;

  // 找到所有子分类
  const children = categories.filter(c => c.parent === parent._id);
  sortSubCategories(children, parent.name);

  // 构建支持三级的子分类数据
  const childrenData = children.map(c => buildLevel2Data(c, categories));

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

  // 按标题名称排序
  const allPosts = Array.from(postMap.values());
  allPosts.sort((a, b) => comparePosts(a.post, b.post));

  return {
    name: parent.name,
    icon: categoryIcons[parent.name] || 'fa-folder-open',
    description: categoryDescriptions[parent.name] || '',
    children: childrenData,
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
        const level3Children = categories.filter(c => c.parent === child._id);
        sortLevel3Categories(level3Children, child.name);
        result += `<li class="category-tree-child${isLast ? ' last' : ''}">`;
        result += `<span class="category-tree-branch">${isLast ? '└──' : '├──'}</span>`;
        result += `<a href="${url}" class="category-tree-link">${child.name}</a>`;
        result += `<span class="category-tree-child-count">${child.length}</span>`;
        if (level3Children.length > 0) {
          result += '<ul class="category-tree-grandchildren">';
          level3Children.forEach((grandchild, gIdx) => {
            const isGLast = gIdx === level3Children.length - 1;
            const gUrl = this.url_for(grandchild.path);
            result += `<li class="category-tree-grandchild${isGLast ? ' last' : ''}">`;
            result += `<span class="category-tree-branch">${isGLast ? '└──' : '├──'}</span>`;
            result += `<a href="${gUrl}" class="category-tree-link">${grandchild.name}</a>`;
            result += `<span class="category-tree-child-count">${grandchild.length}</span>`;
            result += '</li>';
          });
          result += '</ul>';
        }
        result += '</li>';
      });
      result += '</ul>';
    }

    result += '</div>';
  });

  result += '</div>';
  return result;
});

// 自定义首页 generator：排除折叠分类的文章，覆盖 hexo-generator-index
const pagination = require('hexo-pagination');

hexo.extend.generator.register('index', function(locals) {
  const config = this.config;
  const posts = locals.posts.sort(config.index_generator.order_by).filter(post => {
    if (!post.categories || !post.categories.length) return true;
    return !post.categories.toArray().some(cat =>
      homepageCollapsedCategories.includes(cat.name)
    );
  });

  posts.data.sort((a, b) => (b.sticky || 0) - (a.sticky || 0));

  const paginationDir = config.pagination_dir || 'page';
  const path = config.index_generator.path || '';

  return pagination(path, posts, {
    perPage: config.index_generator.per_page,
    layout: ['index', 'archive'],
    format: paginationDir + '/%d/',
    data: {
      __index: true
    }
  });
});
