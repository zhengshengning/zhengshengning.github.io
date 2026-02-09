'use strict';

/**
 * 自定义分类排序脚本
 * 让指定分类（如"计算机基础"）在分类列表中排在第一位
 */

hexo.extend.helper.register('list_categories_custom', function(categories, options) {
  const opts = Object.assign({
    priorityCategory: '计算机基础'
  }, options);

  if (!categories || !categories.length) {
    return '';
  }

  // 转换为数组并排序
  const categoryArray = categories.toArray();
  
  categoryArray.sort((a, b) => {
    // 优先分类排在最前面
    if (a.name === opts.priorityCategory) return -1;
    if (b.name === opts.priorityCategory) return 1;
    // 其他按名称排序
    return a.name.localeCompare(b.name, 'zh-CN');
  });

  return categoryArray;
});

// 在生成前对分类进行排序处理
hexo.extend.filter.register('before_generate', function() {
  const priorityCategory = '计算机基础';
  
  // 对所有分类添加排序权重
  this.locals.get('categories').forEach(function(category) {
    if (category.name === priorityCategory) {
      category._sortOrder = 0; // 最高优先级
    } else {
      category._sortOrder = 1;
    }
  });
});
