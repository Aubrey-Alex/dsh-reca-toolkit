const $ = (s) => document.querySelector(s);
const escapeHtml = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function boot() {
  const data = await fetch('data/replay_manifest.json').then(r => r.json());
  const dsh = data.dsh || {}, reca = data.reca || {};
  $('#run-id').textContent = `RUN ${data.run_id}`;
  $('#footer-run').textContent = data.run_id;
  $('#run-state').textContent = String(dsh.result_state || 'RECORDED').toUpperCase();
  $('#story').textContent = dsh.user_story || 'Recorded user request';
  const metrics = [['SHOT',reca.shot_count],['SEGMENTS',reca.segment_count],['ASSETS',reca.asset_count || '—'],['AUDIT',reca.audit_state || '—']];
  $('#metrics').innerHTML = metrics.map(([a,b]) => `<div class="metric"><b>${escapeHtml(b)}</b><span>${a}</span></div>`).join('');
  const phases = [['01','Planning','DeepSeek story + shot plan'],['02','Generation','Assets + Wan segments'],['03','Audit','Continuity checkpoint'],['04','Delivery','Concat + artifacts']];
  $('#timeline').innerHTML = phases.map(p => `<div class="timeline-step"><div class="timeline-dot">${p[0]}</div><strong>${p[1]}</strong><small>${p[2]}</small></div>`).join('');
  $('#shots').innerHTML = (data.shots || []).map((s,i) => `<article class="shot"><span class="shot-index">SHOT ${String(i+1).padStart(2,'0')}</span><h3>${escapeHtml(s.id || 'Untitled shot')}</h3><p>${escapeHtml(s.start_state || '')}</p><div class="states">NEXT <b>${escapeHtml(s.end_state || '')}</b></div></article>`).join('');
  const auditState = reca.audit_state || 'audit_pending'; $('#audit-state').textContent = auditState.replaceAll('_',' ').toUpperCase();
  $('#audit-title').textContent = auditState === 'audited' ? 'Continuity verified' : 'Visual audit recorded';
  $('#audit-copy').textContent = auditState === 'audit_repaired' ? 'GPT Vision found continuity drift and ReCA repaired the affected segment before delivery.' : 'ReCA checks identity, scene, props and motion between generated segments before delivery.';
  const bundlePath = (kind, path) => kind === 'final_video' ? 'assets/final.mp4' : 'data/replay_manifest.json';
  $('#artifacts').innerHTML = Object.entries(data.artifacts || {}).map(([kind,path]) => `<div class="artifact"><span>${escapeHtml(kind)}</span><a href="${bundlePath(kind, path)}">${escapeHtml(path)}</a></div>`).join('');
  const video = $('#final-video'); video.src = 'assets/final.mp4';
  video.addEventListener('loadedmetadata', () => { $('#video-meta').textContent = `${video.videoWidth} × ${video.videoHeight} · ${video.duration.toFixed(1)}s`; });
}
boot().catch(err => { document.body.dataset.error = err.message; });
