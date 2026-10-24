(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const variants = ['st', 'adobe', 'a7look'];
  const labels = {st: 'Sony ST', adobe: 'Adobe HNCS', a7look: '我们的 a7look · 软件模拟'};
  let photos = [], shown = [], selected = '', category = '全部';
  const byId = id => photos.find(photo => photo.id === id);

  function render() {
    const photo = byId(selected);
    if (!photo) return;
    $('title').textContent = photo.title;
    $('category').textContent = photo.category;
    $('note').textContent = photo.note;
    $('counter').textContent = `${String(shown.findIndex(p => p.id === selected) + 1).padStart(2, '0')} / ${String(shown.length).padStart(2, '0')}`;
    for (const variant of variants) {
      $('image-' + variant).src = photo.images[variant].path;
      $('image-' + variant).alt = `${photo.title} · ${labels[variant]}`;
    }
    for (const button of $('thumbnails').children) {
      button.setAttribute('aria-pressed', String(button.dataset.id === selected));
    }
    if (location.hash !== '#' + selected) history.replaceState(null, '', '#' + selected);
    const following = shown[(shown.findIndex(p => p.id === selected) + 1) % shown.length];
    for (const variant of variants) {const preload = new Image(); preload.src = following.images[variant].path;}
  }

  function select(id) {if (byId(id)) {selected = id; render();}}

  function buildThumbnails() {
    shown = photos.filter(photo => category === '全部' || photo.category === category);
    $('thumbnails').replaceChildren();
    for (const photo of shown) {
      const button = document.createElement('button');
      button.className = 'thumbnail'; button.dataset.id = photo.id;
      button.setAttribute('aria-label', '查看' + photo.title);
      const image = document.createElement('img');
      image.src = photo.images.a7look.path; image.alt = ''; image.loading = 'lazy'; image.width = 132; image.height = 88;
      const title = document.createElement('span'); title.textContent = photo.title;
      button.append(image, title); button.onclick = () => select(photo.id);
      $('thumbnails').appendChild(button);
    }
    if (!shown.some(photo => photo.id === selected)) selected = shown[0].id;
    for (const button of $('categories').children) button.setAttribute('aria-pressed', String(button.dataset.category === category));
    render();
  }

  function move(direction) {
    if (!shown.length) return;
    select(shown[(shown.findIndex(photo => photo.id === selected) + direction + shown.length) % shown.length].id);
  }

  async function start() {
    try {
      const response = await fetch('comparison.json');
      if (!response.ok) throw new Error('照片清单读取失败');
      const manifest = await response.json(); photos = manifest.photos;
      if (photos.length !== 9 || photos.some(photo => variants.some(v => !photo.images[v]?.path))) throw new Error('照片清单不完整');
      selected = byId(location.hash.slice(1)) ? location.hash.slice(1) : photos[0].id;
      buildThumbnails();
      $('previous').onclick = () => move(-1); $('next').onclick = () => move(1);
      for (const button of $('categories').children) button.onclick = () => {category = button.dataset.category; buildThumbnails();};
      for (const button of document.querySelectorAll('.image-button')) button.onclick = () => {
        const variant = button.dataset.variant, photo = byId(selected);
        $('lightbox-image').src = photo.images[variant].path;
        $('lightbox-image').alt = photo.title + ' · ' + labels[variant];
        $('lightbox-label').textContent = photo.title + ' / ' + labels[variant];
        $('lightbox').showModal();
      };
      $('close-lightbox').onclick = () => $('lightbox').close();
      $('lightbox').onclick = event => {if (event.target === $('lightbox')) $('lightbox').close();};
      document.addEventListener('keydown', event => {
        if ($('lightbox').open || ['INPUT','SELECT','TEXTAREA'].includes(event.target.tagName)) return;
        if (event.key === 'ArrowLeft') {event.preventDefault(); move(-1);}
        if (event.key === 'ArrowRight') {event.preventDefault(); move(1);}
      });
      window.addEventListener('hashchange', () => {
        const id = location.hash.slice(1); if (!byId(id) || id === selected) return;
        selected = id;
        if (!shown.some(photo => photo.id === id)) {category = '全部'; buildThumbnails();} else render();
      });
    } catch (error) {
      $('title').textContent = '预览暂时无法载入';
      $('error').hidden = false; $('error').textContent = error.message + '。请通过 GitHub Pages 或本地 HTTP 服务打开此页面。';
    }
  }
  start();
})();
