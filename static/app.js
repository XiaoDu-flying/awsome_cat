const tabs = document.querySelectorAll('.tab');
const panels = document.querySelectorAll('.panel');

tabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    tabs.forEach((item) => item.classList.remove('active'));
    panels.forEach((panel) => panel.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.target).classList.add('active');
  });
});

function setLoading(id, text) {
  document.getElementById(id).innerHTML = `<div class="loading-state">${text}</div>`;
}

async function fetchWithTimeout(url, options = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

function renderCatResult(data) {
  document.getElementById('cat-result').innerHTML = `
    <h3 class="result-title">${escapeHtml(data.title)}</h3>
    <div class="pill-row">
      <span class="pill">判定：${escapeHtml(data.legacy_alignment)}</span>
      <span class="pill">猫缘分：${escapeHtml(String(data.fortune_score))} / 100</span>
    </div>
    <div class="metric"><strong>外貌特点：</strong>${data.appearance.map(escapeHtml).join('、')}</div>
    <div class="metric"><strong>古风点评：</strong>${escapeHtml(data.summary)}</div>
    <div class="tag-row">${data.tags.map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join('')}</div>
  `;
}

function renderDateResult(data) {
  document.getElementById('date-result').innerHTML = `
    <h3 class="result-title">${escapeHtml(data.date)} · ${escapeHtml(data.level)}</h3>
    <div class="pill-row">
      <span class="pill">${data.is_lucky ? '宜纳猫' : '今日暂缓纳猫'}</span>
      <span class="pill">黄历评分：${escapeHtml(String(data.score))}</span>
    </div>
    <div class="metric"><strong>黄历总述：</strong>${escapeHtml(data.summary)}</div>
    <div class="metric"><strong>签语：</strong>${escapeHtml(data.oracle)}</div>
    <div class="columns">
      <div class="metric">
        <strong>宜</strong>
        <ul>${data.yi.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
      </div>
      <div class="metric">
        <strong>忌</strong>
        <ul>${data.ji.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>
      </div>
    </div>
  `;
}

function renderContractResult(data) {
  document.getElementById('contract-result').innerHTML = `
    <h3 class="result-title">${escapeHtml(data.title)}</h3>
    <div class="pill-row">
      <span class="pill">${escapeHtml(data.body_source_label || (data.ai_generated ? '本契正文由 AI 生成' : '本契正文由模板生成'))}</span>
    </div>
    <div class="metric"><strong>契书正文：</strong>${escapeHtml(data.body)}</div>
    <img class="poster-preview" src="${data.image_url}" alt="纳猫契预览" />
    <div style="text-align:center;">
      <a class="action-link" href="${data.image_url}" download="${escapeHtml(data.download_name)}">下载纳猫契图片</a>
    </div>
  `;
}

document.getElementById('cat-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  setLoading('cat-result', '正在翻阅《相猫歌诀》并观察猫相，请稍候…');
  try {
    const response = await fetch('/api/cat/read', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '相猫失败');
    renderCatResult(data);
  } catch (error) {
    document.getElementById('cat-result').innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
});

document.getElementById('date-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  setLoading('date-result', '正在查阅电子黄历，请稍候…');
  try {
    const response = await fetchWithTimeout('/api/date/lucky', { method: 'POST', body: formData }, 60000);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '查询失败');
    renderDateResult(data);
  } catch (error) {
    const message = error.name === 'AbortError'
      ? '前端在 60 秒内未收到 /api/date/lucky 的响应：模型生成时间过长、网络较慢，或后端服务异常。'
      : error.message;
    document.getElementById('date-result').innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
  }
});

document.getElementById('contract-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const formData = new FormData(event.currentTarget);
  setLoading('contract-result', '正在书写纳猫契并加盖印章，请稍候…');
  try {
    const response = await fetch('/api/contract/generate', { method: 'POST', body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '生成失败');
    renderContractResult(data);
  } catch (error) {
    document.getElementById('contract-result').innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
});

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}
