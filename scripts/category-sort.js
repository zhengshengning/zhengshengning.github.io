'use strict';

/**
 * 自定义分类排序脚本
 * 支持两级树状分类结构展示
 */

// 一级分类的展示顺序
const topLevelOrder = ['AI Infra', '战胜玩AI', '编程技能包'];

// 一级分类的图标映射
const categoryIcons = {
  'AI Infra': 'fa-server',
  '战胜玩AI': 'fa-robot',
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
    children.sort((a, b) => b.length - a.length);
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

    // 按文章数降序排列子分类
    children.sort((a, b) => b.length - a.length);

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
