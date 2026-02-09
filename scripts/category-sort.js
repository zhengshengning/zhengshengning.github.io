'use strict';

/**
 * 自定义分类排序脚本
 * 让指定分类（如"计算机基础"）在分类列表中排在第一位
 */

const priorityCategory = '计算机基础';

// 注册一个 helper 来获取排序后的分类数组
hexo.extend.helper.register('sorted_categories', function() {
  const categories = this.site.categories.toArray();
  
  categories.sort((a, b) => {
    // 优先分类排在最前面
    if (a.name === priorityCategory) return -1;
    if (b.name === priorityCategory) return 1;
    // 其他按名称排序
    return a.name.localeCompare(b.name, 'zh-CN');
  });

  return categories;
});

// 注册自定义的 list_categories helper，替代默认的
hexo.extend.helper.register('list_categories_sorted', function(options) {
  const categories = this.site.categories.toArray();
  
  if (!categories || categories.length === 0) {
    return '';
  }

  // 排序：优先分类在前
  categories.sort((a, b) => {
    if (a.name === priorityCategory) return -1;
    if (b.name === priorityCategory) return 1;
    return a.name.localeCompare(b.name, 'zh-CN');
  });

  // 生成 HTML 列表
  let result = '<ul class="category-list">';
  
  categories.forEach(cat => {
    const url = this.url_for(cat.path);
    result += `<li class="category-list-item">`;
    result += `<a class="category-list-link" href="${url}">${cat.name}</a>`;
    result += `<span class="category-list-count">${cat.length}</span>`;
    result += `</li>`;
  });
  
  result += '</ul>';
  return result;
});
