import { writeFileSync, appendFileSync, readFileSync, openSync, closeSync, readSync, writeSync, fsyncSync, statSync, mkdirSync, copyFileSync, renameSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const config = JSON.parse(process.env.ISAAC_TRAINING_CONFIG);
const offlineRoot = join(config.research, 'offline');
const { readObservation, actionToKeyDeltas, TRAINING_OBSERVER_OFFSETS } = await import(pathToFileURL(join(offlineRoot, 'training_observer.mjs')).href);
const { createCorePolicy } = await import(pathToFileURL(join(offlineRoot, 'core_policy.mjs')).href);
const { stateHash } = await import(pathToFileURL(join(offlineRoot, 'canonical_state.mjs')).href);
const { toPlannerFrame } = await import(pathToFileURL(join(offlineRoot, 'planner_frame.mjs')).href);
const { isCooldownHit } = await import(pathToFileURL(join(offlineRoot, 'damage_tracking.mjs')).href);
const staticMechanisms = JSON.parse(readFileSync(
  join(config.research, '..', 'data', 'npc', 'engine_monster_static_mechanism_candidates.json'),
  'utf8',
));
const staticContactDamageByKey = new Map((staticMechanisms.entries ?? []).map(entry => [
  entry.key,
  Number(entry.static_entity_data?.[0]?.collision_damage ?? 0),
]));
const driver = config.engineDriver ?? join(config.research, 'the-browsing-of-isaac/output/recomp/lift/boot-training-fast/boot_integration.mjs');
const { KEYS, makeConsole } = await import(pathToFileURL(join(config.research, 'the-browsing-of-isaac/scripts/recomp/lift/explore.mjs')).href);
let module, active = false, resume = null, pendingStep = null, previousAction = null;
let frame = null, steps = 0, terminal = false;
let rollout = null;
let setupConsole = null, setupConsoleTick = 0, setupTargetSeen = false, setupObservation = null, setupStartedFrame = -1;
let setupIsolationPasses = 0;
let lastSetupTraceFrame = -1000;
let forcedDifficultyApplied = false;
let probe = null;
let warmReset = null;
const policy = await createCorePolicy();
const events = [];
const send = message => { if (process.connected) process.send(message); };
function exportedFS() {
  // This compiled training module does not export Emscripten FS. Accessing its
  // generated `FS` getter aborts the guest, so snapshots cover linear memory
  // plus worker input state. Lazy archive reads use positional reads (pread).
  return null;
}
const trainingSnapshots = new Map();
const trainingForks = new Map();
let nextTrainingSnapshotId = 1;
let nextTrainingForkId = 1;

function trainingBoundaryIsIdle() {
  return active && !!resume && !pendingStep && !rollout && !setupConsole && !probe && !warmReset;
}

function snapshotTrainingState(reuseId = null) {
  if (!trainingBoundaryIsIdle()) throw new Error('snapshot requires an idle, ready game-frame boundary');
  const observation = readObservation(module);
  if (!observation.ready || observation.dead || observation.cleared) {
    throw new Error('snapshot requires a live, uncleared room');
  }
  if (trainingSnapshots.size >= 2 && !trainingSnapshots.has(reuseId)) {
    throw new Error('at most two memory snapshots may be held per instance');
  }
  const fs = exportedFS();
  const streams = (fs?.streams ?? []).filter(Boolean).map(stream => ({
    stream, fd: stream.fd, path: stream.path, position: stream.position, flags: stream.flags,
  }));
  let memory = trainingSnapshots.get(reuseId)?.memory;
  if (!memory || memory.length !== module.HEAPU8.length) memory = new Uint8Array(module.HEAPU8.length);
  memory.set(module.HEAPU8);
  const id = reuseId ?? nextTrainingSnapshotId++;
  const snapshot = {
    id, memory, frame, steps, terminal,
    memoryBytes: memory.byteLength,
    previousAction: structuredClone(previousAction), events: structuredClone(events), streams,
    seed: config.seed, room: config.room, targetNPC: config.targetNPC,
    observationHash: stateHash(observation),
    memorySha256: createHash('sha256').update(memory).digest('hex'),
  };
  trainingSnapshots.set(id, snapshot);
  return {
    id, bytes: memory.byteLength, memoryBytes: memory.byteLength, frame, steps, seed: config.seed, room: config.room,
    targetNPC: config.targetNPC, observationHash: snapshot.observationHash,
    memorySha256: snapshot.memorySha256,
    streams: streams.map(({ stream, ...metadata }) => metadata),
  };
}

function safeSnapshotPath(pathValue) {
  const root = resolve(config.snapshotRoot ?? config.run);
  const path = resolve(pathValue);
  const rel = relative(root, path);
  if (!rel || rel === '..' || rel.startsWith(`..${sep}`) || isAbsolute(rel)) {
    throw new Error('snapshot file must be inside the configured snapshot root');
  }
  return path;
}

function exportTrainingSnapshot(id, pathValue) {
  if (!trainingBoundaryIsIdle()) throw new Error('snapshot export requires an idle game-frame boundary');
  const snapshotId = resolveTrainingSnapshotId(id);
  const snapshot = trainingSnapshots.get(snapshotId);
  const path = safeSnapshotPath(pathValue);
  mkdirSync(dirname(path), { recursive: true });
  if (snapshot.memory) {
    const fd = openSync(path, 'w');
    try {
      for (let offset = 0; offset < snapshot.memory.byteLength;) {
        const length = Math.min(32 * 1024 * 1024, snapshot.memory.byteLength - offset);
        const written = writeSync(fd, snapshot.memory, offset, length, offset);
        if (written <= 0) throw new Error(`snapshot write stalled at ${offset}`);
        offset += written;
      }
      fsyncSync(fd);
    } finally {
      closeSync(fd);
    }
  } else if (snapshot.filePath && snapshot.filePath !== path) {
    copyFileSync(snapshot.filePath, path);
  } else if (snapshot.filePath === path) {
    // Re-exporting the same file is already complete.
  } else {
    throw new Error('snapshot has no memory payload');
  }
  const metadata = {
    schema: 'gamebot-wasm-snapshot/1',
    sourceExecutableSha256: config.sourceExecutableSha256,
    bootWasmSha256: config.bootWasmSha256,
    memoryBytes: snapshot.memoryBytes,
    memorySha256: snapshot.memorySha256,
    frame: snapshot.frame,
    steps: snapshot.steps,
    terminal: snapshot.terminal,
    seed: snapshot.seed,
    room: snapshot.room,
    targetNPC: snapshot.targetNPC,
    previousAction: snapshot.previousAction,
    events: snapshot.events,
    observationHash: snapshot.observationHash,
    streams: snapshot.streams.map(({ stream, ...row }) => row),
  };
  writeFileSync(`${path}.json`, `${JSON.stringify(metadata, null, 2)}\n`);
  return { snapshotId, path, metadataPath: `${path}.json`, bytes: snapshot.memoryBytes, memorySha256: snapshot.memorySha256 };
}

function importTrainingSnapshot(pathValue) {
  if (!trainingBoundaryIsIdle()) throw new Error('snapshot import requires an idle game-frame boundary');
  if (trainingSnapshots.size >= 2) throw new Error('at most two memory snapshots may be held per instance');
  const path = safeSnapshotPath(pathValue);
  const metadata = JSON.parse(readFileSync(`${path}.json`, 'utf8'));
  if (metadata.schema !== 'gamebot-wasm-snapshot/1') throw new Error('unsupported snapshot file schema');
  if (metadata.sourceExecutableSha256 !== config.sourceExecutableSha256 ||
      metadata.bootWasmSha256 !== config.bootWasmSha256) {
    throw new Error('snapshot game/runtime build does not match this worker');
  }
  if (metadata.memoryBytes !== module.HEAPU8.byteLength || statSync(path).size !== metadata.memoryBytes) {
    throw new Error('snapshot WASM memory size does not match this worker');
  }
  if (metadata.streams?.length) throw new Error('cross-process snapshot has unportable open FS streams');
  if (!Array.isArray(metadata.events) || metadata.seed == null || metadata.room == null) {
    throw new Error('snapshot metadata is incomplete');
  }
  const id = nextTrainingSnapshotId++;
  const snapshot = {
    id,
    memory: null,
    filePath: path,
    memoryBytes: metadata.memoryBytes,
    memorySha256: metadata.memorySha256,
    frame: metadata.frame,
    steps: metadata.steps,
    terminal: metadata.terminal,
    seed: metadata.seed,
    room: metadata.room,
    targetNPC: metadata.targetNPC,
    previousAction: metadata.previousAction,
    events: metadata.events,
    streams: [],
    observationHash: metadata.observationHash,
  };
  trainingSnapshots.set(id, snapshot);
  try {
    const result = restoreTrainingState(id);
    return {
      ...result,
      memoryBytes: snapshot.memoryBytes,
      memorySha256: snapshot.memorySha256,
      filePath: path,
      seed: snapshot.seed,
      room: snapshot.room,
      targetNPC: snapshot.targetNPC,
    };
  } catch (error) {
    trainingSnapshots.delete(id);
    throw error;
  }
}

function resolveTrainingSnapshotId(id) {
  if (trainingSnapshots.has(id)) return id;
  const branch = trainingForks.get(id);
  if (branch && trainingSnapshots.has(branch.snapshotId)) return branch.snapshotId;
  throw new Error(`unknown snapshot/fork handle: ${id}`);
}

function restoreTrainingState(id) {
  if (!active || !resume || pendingStep || rollout || setupConsole || probe || warmReset) {
    throw new Error('restore requires an idle, ready game-frame boundary');
  }
  const snapshotId = resolveTrainingSnapshotId(id);
  const snapshot = trainingSnapshots.get(snapshotId);
  const fs = exportedFS();
  for (const saved of snapshot.streams) {
    if (fs?.getStream && fs.getStream(saved.fd) !== saved.stream) {
      throw new Error(`snapshot stream lifetime changed: ${saved.fd} ${saved.path}`);
    }
  }
  const memoryBytes = snapshot.memory?.byteLength ?? snapshot.memoryBytes;
  if (!Number.isInteger(memoryBytes) || module.HEAPU8.length < memoryBytes) {
    throw new Error('snapshot memory is larger than the current WASM heap');
  }
  if (fs?.streams) {
    const savedFds = new Set(snapshot.streams.map(stream => stream.fd));
    for (const stream of fs.streams.filter(Boolean)) {
      if (!savedFds.has(stream.fd)) fs.close(stream);
    }
    for (const saved of snapshot.streams) saved.stream.position = saved.position;
  }
  if (snapshot.memory) {
    module.HEAPU8.set(snapshot.memory);
  } else if (snapshot.filePath) {
    const fd = openSync(snapshot.filePath, 'r');
    const digest = createHash('sha256');
    try {
      for (let offset = 0; offset < memoryBytes;) {
        const length = Math.min(32 * 1024 * 1024, memoryBytes - offset);
        const count = readSync(fd, module.HEAPU8, offset, length, offset);
        if (count !== length) throw new Error(`snapshot file read stalled at ${offset}`);
        digest.update(Buffer.from(module.HEAPU8.buffer, module.HEAPU8.byteOffset + offset, count));
        offset += count;
      }
    } finally {
      closeSync(fd);
    }
    if (digest.digest('hex') !== snapshot.memorySha256) throw new Error('snapshot file memory hash mismatch');
  } else {
    throw new Error('snapshot memory payload is unavailable');
  }
  if (module.HEAPU8.length > memoryBytes) module.HEAPU8.fill(0, memoryBytes);
  frame = snapshot.frame;
  steps = snapshot.steps;
  terminal = snapshot.terminal;
  previousAction = structuredClone(snapshot.previousAction);
  events.splice(0, events.length, ...structuredClone(snapshot.events));
  config.seed = snapshot.seed;
  config.room = snapshot.room;
  config.targetNPC = snapshot.targetNPC;
  const observation = readObservation(module);
  const observationHash = stateHash(observation);
  if (observationHash !== snapshot.observationHash) {
    throw new Error(`snapshot restore observation mismatch at frame ${observation.frame}`);
  }
  return { observation, snapshotId, frame, steps, observationHash };
}

function forkTrainingState(snapshotId, label = '') {
  const id = resolveTrainingSnapshotId(snapshotId);
  const snapshot = trainingSnapshots.get(id);
  const branchId = `fork-${nextTrainingForkId++}`;
  const branch = { id: branchId, snapshotId: id, label: String(label).slice(0, 120), frame: snapshot.frame };
  trainingForks.set(branchId, branch);
  return { ...branch, seed: snapshot.seed, room: snapshot.room, memoryBytes: snapshot.memoryBytes };
}

function dropTrainingState(id) {
  if (trainingForks.delete(id)) return true;
  if (!trainingSnapshots.delete(id)) return false;
  for (const [branchId, branch] of trainingForks) {
    if (branch.snapshotId === id) trainingForks.delete(branchId);
  }
  return true;
}
const isAlive = e => e.exists!==false && e.dead!==true && e.active!==false;
const isEnemy = e => e.type>=10 && e.type<1000;
function contactDamageFor(e) {
  const staticDamage = isEnemy(e) ? staticContactDamageByKey.get(entityIdentity(e)) ?? 0 : 0;
  return {
    damage: e.damage > 0 ? e.damage : staticDamage,
    source: e.damage > 0
      ? 'runtime-entity-damage'
      : staticDamage > 0
        ? 'static-entity-config'
        : null,
  };
}
const targetSpec = () => typeof config.targetNPC==='object'?config.targetNPC:{type:config.targetNPC};
const targetSpecs = () => Array.isArray(config.targetNPCs)?config.targetNPCs:[targetSpec()];
const hasSpawnRequest = () => config.spawnNPCs?.length>0||config.spawnNPC!==undefined;
function isTarget(entity,target=targetSpec()) {
  return isAlive(entity)&&entity.type===target.type&&
    (target.variant===undefined||entity.variant===target.variant)&&
    (target.subtype===undefined||entity.subtype===target.subtype||
      config.allowVariantFallback===true);
}

function allTargetsSeen(observation) {
  return targetSpecs().every(target => observation.entities.some(entity => isTarget(entity,target)));
}

function consoleMemory() {
  return {
    ok: address => Number.isInteger(address)&&address>=0&&address+4<=module.HEAPU8.byteLength,
    u32: address => new DataView(module.HEAPU8.buffer).getUint32(address,true),
    u8: address => module.HEAPU8[address],
    f32: address => new DataView(module.HEAPU8.buffer).getFloat32(address,true),
  };
}

function explosiveNpc(e) {
  return ((e.type===25||e.type===250||e.type===292||e.type===295)&&e.variant===0) ||
    (e.type===55&&(e.variant===1||e.variant===2)) ||
    (e.type===16&&e.variant===2) || e.champion===5;
}

function entityHazards(e) {
  if (!isAlive(e) || e.friendly) return [];
  const kinds = [];
  const projectileFlags = (BigInt(e.projectile_flags_hi ?? 0)<<32n) | BigInt(e.projectile_flags_lo ?? 0);
  if (e.type===9 && !(projectileFlags & (1n<<31n))) kinds.push('projectile');
  if (e.type===2 && (e.hostile_source || (e.spawner_type>=10 && e.spawner_type<1000))) kinds.push('hostile-tear');
  if (e.type===7) kinds.push('laser');
  if (e.type===4) kinds.push('bomb');
  const contact = contactDamageFor(e);
  if (isEnemy(e) && e.collision!==0 && contact.damage>0) kinds.push('contact');
  if (isEnemy(e) && explosiveNpc(e)) kinds.push('explosive-npc');
  if (e.type===1000 && ((e.variant>=22 && e.variant<=26) || e.variant===56)) kinds.push('creep');
  if (e.type===1000 && (e.variant===4 || e.variant===62)) kinds.push('cage-rock-wave');
  if (e.type===1000 && (e.variant===31 || e.variant===19)) kinds.push('falling-impact');
  if (e.type===20 && e.variant===0 && e.collision===0 &&
      (e.animation==='walk' || e.animation==='jumpdown')) kinds.push('monstro-landing');
  return kinds;
}

function hazardSnapshot(plannerFrame) {
  const player = plannerFrame.player.body;
  const hazards = [];
  for (const e of plannerFrame.entities) {
    const kinds = entityHazards(e);
    if (!kinds.length) continue;
    const contact = contactDamageFor(e);
    const d = Math.hypot(e.pos.x-player.pos.x, e.pos.y-player.pos.y);
    if (d>320) continue;
    const item = {
      kinds, address:e.address, type:e.type, variant:e.variant, subtype:e.subtype,
      position:e.pos, velocity:e.vel, target_pos:e.target_pos, radius:e.radius, damage:e.damage,
      contact_damage:contact.damage, contact_damage_source:contact.source,
      hp:e.hp, invulnerable:e.invulnerable, no_target:e.no_target, targetable:e.targetable,
      collision:e.collision, state:e.state, state_frame:e.state_frame,
      animation:e.animation, animation_frame:e.animation_frame, angle:e.angle,
      spawner:e.spawner, parent:e.parent, distance:d,
    };
    if (e.type===9) {
      const flags=(BigInt(e.projectile_flags_hi ?? 0)<<32n) | BigInt(e.projectile_flags_lo ?? 0);
      item.projectile_flags=flags.toString(16).padStart(16,'0');
    }
    if (e.type===7) Object.assign(item, {
      endpoint:e.endpoint, laser_radius:e.laser_radius, laser_follow:e.laser_follow,
      circle:e.circle, geometry_valid:e.geometry_valid, timeout:e.timeout,
    });
    if (e.blast_radius) item.blast_radius=e.blast_radius;
    hazards.push(item);
  }
  for (let index = 0; index < plannerFrame.grid.cells.length; index++) {
    const cell = plannerFrame.grid.cells[index];
    if (!cell.danger) continue;
    const x = (index % plannerFrame.grid.width) * 40 + 40;
    const y = Math.floor(index / plannerFrame.grid.width) * 40 + 120;
    const d = Math.hypot(x - player.pos.x, y - player.pos.y);
    if (d > 320) continue;
    hazards.push({
      kinds: [cell.type === 20 ? 'greed-button' : 'grid-danger'],
      address: null,
      type: cell.type,
      variant: cell.variant,
      subtype: 0,
      position: { x, y },
      velocity: { x: 0, y: 0 },
      target_pos: { x: 0, y: 0 },
      radius: 20,
      damage: cell.type === 20 ? 1 : null,
      hp: 0,
      invulnerable: true,
      no_target: false,
      targetable: false,
      collision: cell.collision,
      state: cell.state,
      state_frame: 0,
      animation: '',
      animation_frame: 0,
      angle: 0,
      spawner: 0,
      parent: 0,
      distance: d,
    });
  }
  hazards.sort((a,b)=>a.distance-b.distance);
  return hazards.slice(0,16);
}

function entityIdentity(entity) {
  return `${entity.type}.${entity.variant}.${entity.subtype}`;
}

function traceEntitySource(entity, entitiesByAddress) {
  const chain = [];
  const seen = new Set([entity.address]);
  let cursor = entity;
  for (let depth = 0; depth < 6; depth++) {
    const linked = [
      ['spawner', cursor.spawner],
      ['parent', cursor.parent],
      ['owner', cursor.owner],
    ].find(([, address]) => Number.isInteger(address) && address > 0 && !seen.has(address));
    if (!linked) break;
    const [relation, address] = linked;
    seen.add(address);
    const source = entitiesByAddress.get(address);
    chain.push({
      relation,
      address,
      present: !!source,
      identity: source ? entityIdentity(source) : null,
      type: source?.type ?? null,
      variant: source?.variant ?? null,
      subtype: source?.subtype ?? null,
      friendly: source?.friendly ?? null,
      state: source?.state ?? null,
      animation: source?.animation ?? null,
      animation_frame: source?.animation_frame ?? null,
    });
    if (!source) break;
    cursor = source;
  }
  const hostile = chain.find(source =>
    source.present && source.type >= 10 && source.type < 1000 && source.friendly !== true
  );
  const self = isEnemy(entity) && !entity.friendly ? {
    relation: 'self',
    address: entity.address,
    present: true,
    identity: entityIdentity(entity),
    type: entity.type,
    variant: entity.variant,
    subtype: entity.subtype,
    friendly: entity.friendly,
    state: entity.state,
    animation: entity.animation,
    animation_frame: entity.animation_frame,
  } : null;
  return { chain, source: hostile ?? self };
}

function geometryActor(entity) {
  return {
    address:entity.address,seed:entity.seed,birth:entity.birth,identity:entityIdentity(entity),
    type:entity.type,variant:entity.variant,subtype:entity.subtype,
    position:entity.pos,velocity:entity.vel,target_pos:entity.target_pos,angle:entity.angle,
    radius:entity.radius,damage:entity.damage,collision:entity.collision,hp:entity.hp,
    state:entity.state,state_frame:entity.state_frame,animation:entity.animation,
    animation_frame:entity.animation_frame,parent:entity.parent,spawner:entity.spawner,
    owner:entity.owner,child:entity.child,target:entity.target,
    ai_v1:entity.ai_v1,ai_v2:entity.ai_v2,ai_i1:entity.ai_i1,ai_i2:entity.ai_i2,
    projectile_cooldown:entity.projectile_cooldown,projectile_delay:entity.projectile_delay,
    champion:entity.champion,is_boss:entity.is_boss,invulnerable:entity.invulnerable,
    no_target:entity.no_target,targetable:entity.targetable,
  };
}

function geometryHazard(entity, kinds, entitiesByAddress) {
  const source = traceEntitySource(entity, entitiesByAddress);
  const item = {
    kinds,address:entity.address,seed:entity.seed,birth:entity.birth,identity:entityIdentity(entity),
    type:entity.type,variant:entity.variant,subtype:entity.subtype,
    source_address:source.source?.address ?? null,source_identity:source.source?.identity ?? null,
    source_chain:source.chain,
    position:entity.pos,velocity:entity.vel,target_pos:entity.target_pos,angle:entity.angle,
    radius:entity.radius,damage:entity.damage,collision:entity.collision,hp:entity.hp,
    state:entity.state,state_frame:entity.state_frame,animation:entity.animation,
    animation_frame:entity.animation_frame,parent:entity.parent,spawner:entity.spawner,
    owner:entity.owner,child:entity.child,target:entity.target,
    friendly:entity.friendly,hostile_source:entity.hostile_source,
    geometry_valid:entity.geometry_valid ?? null,
  };
  if (entity.type === 4) Object.assign(item, {
    timeout:entity.timeout,blast_radius:entity.blast_radius,
  });
  if (entity.type === 9) {
    const flags = (BigInt(entity.projectile_flags_hi ?? 0) << 32n) |
      BigInt(entity.projectile_flags_lo ?? 0);
    Object.assign(item, {
      projectile_flags:`0x${flags.toString(16).padStart(16, '0')}`,
      projectile_age:entity.projectile_age,projectile_phase:entity.projectile_phase,
      projectile_curve:entity.projectile_curve,
      projectile_orbit_distance:entity.projectile_orbit_distance,
      height:entity.height,fall_speed:entity.fall_speed,fall_accel:entity.fall_accel,
    });
  }
  if (entity.type === 7) Object.assign(item, {
    endpoint:entity.endpoint,laser_radius:entity.laser_radius,laser_follow:entity.laser_follow,
    laser_points:entity.laser_points ?? [],circle:entity.circle,timeout:entity.timeout,
  });
  if (entity.type === 1000) item.timeout = entity.timeout;
  return item;
}

function flushGeometryTimeline() {
  if (!rollout?.geometryBuffer?.length) return;
  const payload = `${rollout.geometryBuffer.join('\n')}\n`;
  appendFileSync(join(config.run, 'geometry-timeline.jsonl'), payload);
  rollout.geometryBytes += Buffer.byteLength(payload);
  rollout.geometryBuffer = [];
}

function trackGeometryFrame(plannerFrame) {
  if (!rollout?.recordGeometryTimeline || rollout.lastGeometryFrame === plannerFrame.clock) return;
  rollout.lastGeometryFrame = plannerFrame.clock;
  const entitiesByAddress = new Map(plannerFrame.entities.map(entity => [entity.address, entity]));
  const actors = plannerFrame.entities
    .filter(entity => isAlive(entity) && isEnemy(entity) && !entity.friendly)
    .map(geometryActor);
  const hazards = plannerFrame.entities
    .filter(entity => isAlive(entity) && !entity.friendly)
    .flatMap(entity => {
      const kinds = entityHazards(entity);
      return kinds.length ? [geometryHazard(entity, kinds, entitiesByAddress)] : [];
    });
  const row = {
    frame:plannerFrame.clock,
    room:plannerFrame.room,
    player:{
      position:plannerFrame.player.body.pos,
      velocity:plannerFrame.player.body.vel,
      health:plannerFrame.player.health,
    },
    grid:{ width:plannerFrame.grid.width, height:plannerFrame.grid.height },
    actors,
    hazards,
  };
  const line = JSON.stringify(row);
  rollout.geometryBuffer.push(line);
  rollout.geometryFrames++;
  rollout.geometryActorSamples += actors.length;
  rollout.geometryHazardSamples += hazards.length;
  for (const actor of actors) {
    rollout.geometrySourceIdentities.add(actor.identity);
  }
  for (const hazard of hazards) {
    for (const kind of hazard.kinds) {
      rollout.geometryHazardKinds.set(kind, (rollout.geometryHazardKinds.get(kind) ?? 0) + 1);
    }
  }
  if (rollout.geometryBuffer.length >= 60) flushGeometryTimeline();
}

function trackFrame(plannerFrame) {
  if (!rollout || !plannerFrame) return;
  for (const e of plannerFrame.entities) {
    if (!isAlive(e)) continue;
    if (isEnemy(e)) {
      const identity = `${e.type}.${e.variant}.${e.subtype}`;
      const prior = rollout.enemyTypes.get(identity);
      if (prior) {
        prior.lastFrame = plannerFrame.clock;
        if (Number.isFinite(prior.minHp) && Number.isFinite(e.hp) && e.hp < prior.minHp - 1e-6) {
          rollout.enemyHpTransitions.push({
            frame:plannerFrame.clock,
            address:e.address,
            type:e.type,
            variant:e.variant,
            subtype:e.subtype,
            previousHp:prior.minHp,
            hp:e.hp,
            position:e.pos,
            velocity:e.vel,
            collision:e.collision,
            state:e.state,
            stateFrame:e.state_frame,
            aiI1:e.ai_i1,
            aiI2:e.ai_i2,
            deliriumState:e.delirium_state,
            deliriumAttackId:e.delirium_attack_id,
            deliriumCycleId:e.delirium_cycle_id,
            animation:e.animation,
            animationFrame:e.animation_frame,
            decisionFrame:rollout.lastDecision ? {
              move:rollout.lastDecision.move,
              fire:rollout.lastDecision.fire,
              target:rollout.lastDecision.target,
              safe:rollout.lastDecision.safe,
              firingDirection:rollout.lastDecision.analysis.firing_direction,
              firingMiss:rollout.lastDecision.analysis.firing_miss,
              firingTime:rollout.lastDecision.analysis.firing_time,
            } : null,
          });
        }
        if (Number.isFinite(e.hp)) prior.minHp = Math.min(prior.minHp, e.hp);
      } else {
        rollout.enemyTypes.set(identity, {
          type:e.type, variant:e.variant, subtype:e.subtype,
          firstFrame:plannerFrame.clock, lastFrame:plannerFrame.clock,
          minHp:Number.isFinite(e.hp)?e.hp:null,
        });
      }
    }
    if (rollout.recordEntityTypes.has(-1)||rollout.recordEntityTypes.has(e.type)) {
      const signature=`${e.type}|${e.variant}|${e.state}|${e.animation}|${e.birth}|${e.collision}|${e.invulnerable}|${e.no_target}|${e.targetable}`;
      const priorTransition=rollout.lastEntityTransition.get(e.address);
      if(priorTransition!==signature) {
        rollout.lastEntityTransition.set(e.address,signature);
        const transition = {
          frame:plannerFrame.clock,address:e.address,type:e.type,variant:e.variant,subtype:e.subtype,
          position:e.pos,velocity:e.vel,target_pos:e.target_pos,radius:e.radius,collision:e.collision,damage:e.damage,
          hp:e.hp,invulnerable:e.invulnerable,no_target:e.no_target,targetable:e.targetable,
          state:e.state,state_frame:e.state_frame,animation:e.animation,animation_frame:e.animation_frame,
          parent:e.parent,child:e.child,target:e.target,spawner:e.spawner,
          max_hp:e.max_hp,speed_multiplier:e.speed_multiplier,
          projectile_cooldown:e.projectile_cooldown,projectile_delay:e.projectile_delay,
          ai_v1:e.ai_v1,ai_v2:e.ai_v2,ai_i1:e.ai_i1,ai_i2:e.ai_i2,
          is_boss:e.is_boss,shield_strength:e.shield_strength,boss_color:e.boss_color,
          boss_status_cooldown:e.boss_status_cooldown,
          timeout:e.timeout,blast_radius:e.blast_radius,height:e.height,
          fall_speed:e.fall_speed,fall_accel:e.fall_accel,
          projectile_flags:e.projectile_flags,projectile_age:e.projectile_age,
          projectile_phase:e.projectile_phase,projectile_curve:e.projectile_curve,
          projectile_orbit_distance:e.projectile_orbit_distance,
          laser_points:e.laser_points,
          delirium_boss_type:e.delirium_boss_type,delirium_boss_variant:e.delirium_boss_variant,
          delirium_transformation_timer:e.delirium_transformation_timer,
          delirium_remaining_attacks:e.delirium_remaining_attacks,delirium_state:e.delirium_state,
          delirium_attack_id:e.delirium_attack_id,delirium_attack_id_low24:e.delirium_attack_id_low24,
          delirium_attack_angle:e.delirium_attack_angle,delirium_cycle_id:e.delirium_cycle_id,
          delirium_cycle_low7:e.delirium_cycle_low7,
          delirium_cycle_field_15_24:e.delirium_cycle_field_15_24,
          delirium_cycle_field_25_34:e.delirium_cycle_field_25_34,
        };
        if (e.type===7) Object.assign(transition, {
          endpoint:e.endpoint, laser_radius:e.laser_radius, laser_follow:e.laser_follow,
          circle:e.circle, geometry_valid:e.geometry_valid, timeout:e.timeout, angle:e.angle,
        });
        if (e.type===2) Object.assign(transition, {
          playerPosition:plannerFrame.player.body.pos,
          playerVelocity:plannerFrame.player.body.vel,
        });
        rollout.entityTransitions.push(transition);
      }
    }
    const kinds = entityHazards(e);
    for (const kind of kinds) {
      const seen = rollout.dangerFrameKinds.get(plannerFrame.clock);
      if (seen?.has(kind)) continue;
      const bucket = seen ?? new Set();
      bucket.add(kind);
      rollout.dangerFrameKinds.set(plannerFrame.clock, bucket);
      rollout.dangerFrameCounts.set(kind, (rollout.dangerFrameCounts.get(kind) ?? 0) + 1);
    }
  }
  for (const cell of plannerFrame.grid.cells) {
    if (!cell.danger) continue;
    const kind = cell.type === 20 ? 'greed-button' : 'grid-danger';
    const seen = rollout.dangerFrameKinds.get(plannerFrame.clock);
    if (seen?.has(kind)) continue;
    const bucket = seen ?? new Set();
    bucket.add(kind);
    rollout.dangerFrameKinds.set(plannerFrame.clock, bucket);
    rollout.dangerFrameCounts.set(kind, (rollout.dangerFrameCounts.get(kind) ?? 0) + 1);
  }
  rollout.lastFrameHazards=hazardSnapshot(plannerFrame);
  trackGeometryFrame(plannerFrame);
}

function decisionSummary(decision) {
  if (!decision) return null;
  return {
    move:decision.move, fire:decision.fire, target:decision.target,
    valid:decision.valid, safe:decision.safe, stop:decision.stop,
    navigation_boost:decision.navigation_boost,
    clearance:decision.clearance, end_clearance:decision.end_clearance,
    collision_cost:decision.collision_cost,
    planning_ms:decision.planning_ms, threats:decision.threats,
    threat_kinds:decision.threat_kinds, reason:decision.reason, analysis:decision.analysis,
  };
}

function damageDetails(observation, amount, priorHealth, extra = {}) {
  const beforeHazards=rollout?.lastFrameHazards ?? [];
  const plannerFrame = toPlannerFrame(module, observation, false);
  const player = plannerFrame.player.body;
  const hazards=hazardSnapshot(plannerFrame);
  trackFrame(plannerFrame);
  return {
    frame:observation.frame,
    amount,
    priorHealth,
    health:observation.player?.health,
    player:{position:player.pos, velocity:player.vel, radius:player.radius},
    lastDecision:decisionSummary(rollout?.lastDecision),
    beforeHazards,
    hazards,
    hazardCount:hazards.length,
    decisionHistory:rollout?.recordDecisionHistory ? rollout.decisionHistory.slice(-20) : [],
    ...extra,
  };
}

function localGrid(frame) {
  const p=frame.player.body.pos, w=frame.grid.width, h=frame.grid.height;
  const cx=Math.round((p.x-40)/40), cy=Math.round((p.y-120)/40), cells=[];
  for(let y=Math.max(0,cy-2);y<=Math.min(h-1,cy+2);y++)for(let x=Math.max(0,cx-2);x<=Math.min(w-1,cx+2);x++) {
    const c=frame.grid.cells[y*w+x];
    if(c.collision||c.danger)cells.push({x,y,type:c.type,variant:c.variant,state:c.state,collision:c.collision,danger:c.danger});
  }
  return cells;
}

function sameIdentity(entity,target) {
  return entity.type===target.type&&entity.variant===target.variant&&entity.subtype===target.subtype;
}

function spawnIdentityCommand(entry) {
  if(entry.spawnCommand)return entry.spawnCommand;
  const suffix=entry.champion===undefined?'':`.${entry.champion}`;
  return `spawn ${entry.type}.${entry.variant}.${entry.subtype}${suffix}`;
}

function identityCounts(observation) {
  const counts=new Map();
  for(const entity of observation.entities) {
    const key=`${entity.type}.${entity.variant}.${entity.subtype}`;
    counts.set(key,(counts.get(key)??0)+1);
  }
  return counts;
}

function probeTargetCount(observation) {
  const target=probe?.entry;
  if(!target)return 0;
  return observation.entities.filter(entity=>sameIdentity(entity,target)).length;
}

function probeVariantCount(observation) {
  const target=probe?.entry;
  if(!target)return 0;
  return observation.entities.filter(entity=>
    entity.type===target.type&&entity.variant===target.variant).length;
}

function captureProbeTarget(observation, source, allowVariantFallback=false) {
  if(!probe)return;
  const exactMatch=entity=>sameIdentity(entity,probe.entry);
  const match=allowVariantFallback
    ? entity=>entity.type===probe.entry.type&&entity.variant===probe.entry.variant
    : exactMatch;
  const target=observation.entities.find(entity=>match(entity)&&isAlive(entity)) ??
    observation.entities.find(match);
  if(!target)return;
  probe.targetSeen=true;
  probe.entry.observedIdentity=`${target.type}.${target.variant}.${target.subtype}`;
  if(exactMatch(target))probe.entry.identityObserved=true;
  else probe.entry.normalizedIdentityObserved=true;
  probe.entry.firstFrame??=observation.frame;
  probe.entry.lastFrame=observation.frame;
  probe.entry.firstTarget??=target;
  probe.entry.lastTarget=target;
  probe.entry.observedFrames=Math.max(probe.entry.observedFrames??0,1);
  if(Number.isFinite(target.hp)) {
    probe.entry.minHp=Math.min(probe.entry.minHp??target.hp,target.hp);
    probe.entry.maxHp=Math.max(probe.entry.maxHp??target.hp,target.hp);
  }
  try {
    const plannerFrame=toPlannerFrame(module,observation,false);
    const plannerTarget=plannerFrame.entities.find(entity=>entity.address===target.address);
    if(plannerTarget) {
      probe.entry.animations??=[];
      if(plannerTarget.animation&&!probe.entry.animations.includes(plannerTarget.animation))
        probe.entry.animations.push(plannerTarget.animation);
      probe.entry.hazardKinds??=[];
      for(const kind of entityHazards(plannerTarget))
        if(!probe.entry.hazardKinds.includes(kind))probe.entry.hazardKinds.push(kind);
      probe.entry.lastState=plannerTarget.state??null;
      probe.entry.lastAnimation=plannerTarget.animation||null;
    }
  } catch(error) {
    probe.entry.evidenceError??=error.message;
  }
  if(source)probe.entry.firstEvidenceSource??=source;
}

function captureProbeNewIdentities(observation) {
  if(!probe)return;
  probe.entry.newIdentities??=[];
  for(const entity of observation.entities) {
    const key=`${entity.type}.${entity.variant}.${entity.subtype}`;
    if((probe.baseline.get(key)??0)>0||probe.entry.newIdentities.some(row=>row.key===key))continue;
    const observedIdentity={
      key,type:entity.type,variant:entity.variant,subtype:entity.subtype,
      parent:entity.parent,active:entity.active,exists:entity.exists,dead:entity.dead,
      hp:entity.hp,damage:entity.damage,collision:entity.collision,
    };
    probe.entry.newIdentities.push(observedIdentity);
    appendFileSync(join(config.run,'probe-identities.jsonl'),JSON.stringify({
      frame:observation.frame,parentEntry:probe.entry.key,observed:observedIdentity,
    })+'\n');
  }
}

function probeStartConsole(command,phase) {
  events.length=0;
  probe.console=makeConsole(consoleMemory(), {
    commands:Array.isArray(command)?command:[command],
    mode:'type',
    ready:()=>true,
    log:line=>console.log(line),
  });
  probe.consoleTick=0;
  probe.phase=phase;
  probe.startedFrame=frame;
}

function startSetupConsole(commands,phase) {
  setupConsole=makeConsole(consoleMemory(), {
    commands,
    mode:'type',
    ready:()=>true,
    log:line=>console.log(line),
  });
  setupConsoleTick=0;
  setupTargetSeen=false;
  setupObservation=null;
  setupStartedFrame=frame;
  setupConsole.phase=phase;
}

function seedLogState() {
  try {
    const log = readFileSync(join(config.run, 'engine.log'), 'utf8');
    const matches = [...log.matchAll(/RNG Start Seed:\s*([A-Z0-9]{4}) ([A-Z0-9]{4})/g)];
    return { count: matches.length, seed: matches.at(-1)?.slice(1).join(' ') ?? null };
  } catch {
    return { count: 0, seed: null };
  }
}

function probeStartReset() {
  queueAction({moveX:0,moveY:0,fire:'none'});
  probe.phase='release-reset';
}

function probeCompleteEntry(status,error=null) {
  const entry=probe.entry;
  entry.status=status;
  entry.error=error;
  entry.completedAtFrame=frame;
  entry.targetCountGain=(entry.targetCountGain??probe.maxTargetCount)-
    (probe.baseline.get(`${entry.type}.${entry.variant}.${entry.subtype}`)??0);
  probe.results.push(entry);
  appendFileSync(join(config.run,'probe-results.jsonl'),JSON.stringify(entry)+'\n');
  send({event:'probeResult',result:entry});
  probe.index++;
  if(probe.index>=probe.entries.length) {
    const result={results:probe.results,observed:probe.results.filter(row=>row.identityObserved).length};
    send({id:probe.id,result});
    probe=null;
    return;
  }
  const next=probe.entries[probe.index];
  probe.entry={...next,index:probe.index};
  const current=readObservation(module);
  probe.baseline=identityCounts(current);
  probe.baselineVariant=current.entities.filter(entity=>
    entity.type===probe.entry.type&&entity.variant===probe.entry.variant).length;
  probe.targetSeen=false;
  probe.maxTargetCount=0;
  probe.observedFrames=0;
  probe.entry.observedFrames=0;
  probe.phase='release';
  queueAction({moveX:0,moveY:0,fire:'none'});
}

function startProbe(message) {
  module.isaacTrainingNoRender=false;
  policy.configure({dt:1/30,contact_margin:8,...(message.action?.parameters??{})});
  probe={
    id:message.id,
    entries:message.action?.entries??[],
    observeFrames:Math.max(1,message.action?.observeFrames??30),
    attack:message.action?.attack ?? true,
    setupCommands:message.action?.setupCommands??[],
    parameters:{dt:1/30,contact_margin:8,...(message.action?.parameters??{})},
    index:0,
    results:[],
    phase:'release',
    console:null,
    entry:null,
    baseline:null,
    targetSeen:false,
    maxTargetCount:0,
    observedFrames:0,
    startedFrame:frame,
    lastFrame:-1,
  };
  if(!probe.entries.length) {
    send({id:probe.id,result:{results:[],observed:0}});
    probe=null;
    return;
  }
  probe.entry={...probe.entries[0],index:0,observedFrames:0};
  const current=readObservation(module);
  probe.baseline=identityCounts(current);
  probe.baselineVariant=current.entities.filter(entity=>
    entity.type===probe.entry.type&&entity.variant===probe.entry.variant).length;
  probe.entry.startedAtFrame=frame;
  queueAction({moveX:0,moveY:0,fire:'none'});
}

function probePolicyAction(observation) {
  if(!probe)return;
  try {
    const result=policy.decide(module,observation,probe.attack);
    probe.entry.lastDecision=decisionSummary(result.decision);
    probe.entry.policyThreatKinds??=[];
    for(const kind of result.decision.threat_kinds??[])
      if(!probe.entry.policyThreatKinds.includes(kind))probe.entry.policyThreatKinds.push(kind);
    queueAction(result.action);
  } catch(error) {
    probe.entry.policyError=error.message;
  }
}

function handleProbe(observation,roomReady,consoleState) {
  if(!probe)return false;
  if(probe.lastFrame===observation.frame)return true;
  probe.lastFrame=observation.frame;
  frame=observation.frame;
  const count=probeTargetCount(observation);
  const variantCount=probeVariantCount(observation);
  probe.maxTargetCount=Math.max(probe.maxTargetCount??0,count);
  const delta=count-(probe.baseline.get(`${probe.entry.type}.${probe.entry.variant}.${probe.entry.subtype}`)??0);
  if(delta>0)probe.entry.targetCountGain=Math.max(probe.entry.targetCountGain??0,delta);
  if(probe.phase==='release') {
    if(events.length)return true;
    probeStartConsole([
      ...probe.setupCommands,
      spawnIdentityCommand(probe.entry),
    ],'spawning');
    return true;
  }
  if(probe.phase==='spawning') {
    if(delta>0) {
      captureProbeTarget(observation,'spawn',false);
    } else if(!probe.entry.exactOnly&&variantCount>probe.baselineVariant) {
      captureProbeTarget(observation,'spawn',true);
    }
    if(probe.targetSeen&&consoleState===0) {
      probe.console=null;
      probe.phase='observing';
      probe.observedFrames=0;
      probe.entry.observeStartedFrame=observation.frame;
      probe.lastDamageCooldown=observation.player?.damageCooldown;
      probePolicyAction(observation);
    } else if((!probe.targetSeen&&probe.console?.done()&&observation.frame-probe.startedFrame>180)||
              observation.frame-probe.startedFrame>2400) {
      probe.entry.error='spawned identity was not observed';
      probe.console=null;
      probeStartReset();
    }
    return true;
  }
  if(probe.phase==='observing') {
    const hp=observation.player?.health;
    const cooldown=observation.player?.damageCooldown;
    if(isCooldownHit(probe.lastDamageCooldown,cooldown,config.damageCalibration)) {
      probe.entry.hitCount=(probe.entry.hitCount??0)+1;
      probe.entry.hitFrames??=[];
      probe.entry.hitFrames.push(observation.frame);
    }
    probe.lastDamageCooldown=cooldown;
    if(Number.isFinite(hp)) {
      probe.entry.damageTaken=(probe.entry.damageTaken??0)+
        Math.max(0,(probe.entry.lastHealth??hp)-hp);
      probe.entry.lastHealth=hp;
    }
    const live=observation.entities.filter(entity=>sameIdentity(entity,probe.entry)&&isAlive(entity));
    if(live.length)captureProbeTarget(observation,'observation');
    captureProbeNewIdentities(observation);
    probe.observedFrames++;
    probe.entry.observedFrames=probe.observedFrames;
    if(probe.observedFrames>=probe.observeFrames||!live.length) {
      probe.entry.targetAliveAtEnd=live.length>0;
      probe.console=null;
      probeStartReset();
    } else {
      probePolicyAction(observation);
    }
    return true;
  }
  if(probe.phase==='release-reset') {
    if(events.length)return true;
    probeStartConsole(`goto ${config.room}`,'resetting');
    return true;
  }
  if(probe.phase==='resetting') {
    if(probe.console?.done()&&roomReady) {
      probe.console=null;
      const status=probe.entry.identityObserved?'observed':
        probe.entry.normalizedIdentityObserved?'observed-normalized':'spawn-failed';
      probeCompleteEntry(status);
    } else if(observation.frame-probe.startedFrame>2400) {
      probeCompleteEntry('reset-timeout','room did not become ready after goto');
    }
    return true;
  }
  return true;
}

function queueAction(action) {
  const delta = actionToKeyDeltas(action, previousAction);
  if (previousAction === null) {
    const seen = new Set();
    for (const [vk, scan, extended] of Object.values(KEYS)) {
      if (seen.has(vk)) continue;
      seen.add(vk); events.push([1, vk, scan | (extended << 8), 0]);
    }
  }
  events.push(...delta);
  previousAction = action;
}

function greedButton(observation) {
  const cell = observation.grid?.find(candidate =>
    candidate.type === 20 && candidate.variant === 2
  );
  return cell ? {
    cell,
    x: cell.x * 40 + 40,
    y: cell.y * 40 + 120,
  } : null;
}

function greedWaveAssistAction(observation) {
  if (config.greedWaveAssist !== true || !observation.player || !observation.room) return null;
  const assist = rollout.waveAssist;
  if (assist.cooldown > 0) assist.cooldown--;

  if (assist.phase === 'retreat') {
    assist.phaseFrames--;
    if (assist.phaseFrames <= 0) {
      assist.phase = 'idle';
      assist.cooldown = 60;
    }
    return { moveX: 0, moveY: -1, fire: 'none' };
  }

  if ((observation.game?.difficulty | 0) < 2) return null;
  if ((observation.room.type | 0) !== 1 || (observation.room.variant | 0) !== 1018) return null;
  const button = greedButton(observation);
  if (!button) return null;
  const difficulty = observation.game?.difficulty | 0;
  const waveIndex = observation.game?.greedWaveIndex | 0;
  const regularFinalWave = difficulty === 3 ? 9 : 8;
  const timerStopped = (observation.room.greedWaveTimer | 0) === -1;
  const noLivingEnemies =
    ((observation.room.aliveA | 0) + (observation.room.aliveB | 0)) <= 0;
  const initialWaveStart =
    assist.regularStarted !== true &&
    waveIndex === 0 &&
    timerStopped &&
    noLivingEnemies;
  const bossWaveStart =
    assist.regularStarted === true &&
    assist.bossStarted !== true &&
    waveIndex === regularFinalWave &&
    observation.cleared === true &&
    timerStopped &&
    noLivingEnemies;

  if (assist.phase === 'idle') {
    if ((!initialWaveStart && !bossWaveStart) || assist.cooldown > 0) return null;
    if (initialWaveStart) assist.regularStarted = true;
    if (bossWaveStart) assist.bossStarted = true;
    assist.startWaveIndex = waveIndex;
    assist.phase = 'approach';
  }
  if (assist.phase !== 'approach') return null;

  const dx = button.x - observation.player.position.x;
  const dy = button.y - observation.player.position.y;
  const waveActivated =
    (observation.room.greedWaveTimer | 0) !== -1 ||
    waveIndex !== assist.startWaveIndex;
  if (waveActivated) {
    assist.phase = 'retreat';
    assist.phaseFrames = 30;
    return { moveX: 0, moveY: -1, fire: 'none' };
  }
  if (Math.hypot(dx, dy) <= 20) {
    return { moveX: 0, moveY: 0, fire: 'none' };
  }
  const moveX = Math.abs(dx) > 12 ? Math.sign(dx) : 0;
  const moveY = Math.abs(dy) > 12 ? Math.sign(dy) : 0;
  return { moveX, moveY, fire: 'none' };
}

function greedButtonEscapeAction(observation) {
  if (config.greedWaveAssist !== true || !observation.player || !observation.room) return null;
  if ((observation.game?.difficulty | 0) < 2) return null;
  if ((observation.room.type | 0) !== 1 || (observation.room.variant | 0) !== 1018) return null;
  if ((observation.room.greedWaveTimer | 0) < 0) return null;
  const button = greedButton(observation);
  if (!button) return null;

  const dx = observation.player.position.x - button.x;
  const dy = observation.player.position.y - button.y;
  const distance = Math.hypot(dx, dy);
  if (distance >= 56) return null;

  let moveX = 0;
  let moveY = 0;
  if (distance < 1) {
    moveY = -1;
  } else {
    const ax = Math.abs(dx), ay = Math.abs(dy);
    moveX = ax >= ay * 0.38 ? Math.sign(dx) : 0;
    moveY = ay >= ax * 0.38 ? Math.sign(dy) : 0;
    if (moveX === 0 && moveY === 0) moveY = -1;
  }
  rollout.greedButtonEscapeFrames = (rollout.greedButtonEscapeFrames ?? 0) + 1;
  rollout.lastDecision = {
    move: 0,
    fire: 0,
    target: 0,
    valid: true,
    safe: true,
    stop: false,
    reason: 'Greed active switch escape',
  };
  return { moveX, moveY, fire: 'none' };
}

function updateGreedTargetProgress(observation) {
  if (config.greedDropStalledTargets !== true || !observation.room) return;
  if (observation.room.type !== 1 || observation.room.variant !== 1018) return;
  const address = rollout.lastDecision?.target;
  if (!address) return;
  const target = observation.entities.find(entity => entity.address === address);
  if (!target || !isAlive(target)) {
    rollout.targetProgress.delete(address);
    return;
  }
  const hp = Number.isFinite(target.hp) ? target.hp : null;
  const prior = rollout.targetProgress.get(address);
  if (!prior) {
    rollout.targetProgress.set(address, { bestHp: hp, lastProgressFrame: observation.frame });
    return;
  }
  if (hp == null || (Number.isFinite(prior.bestHp) && hp < prior.bestHp - 1e-6)) {
    prior.bestHp = hp;
    prior.lastProgressFrame = observation.frame;
    return;
  }
  if (observation.frame - prior.lastProgressFrame < 600) return;
  rollout.suppressedTargets.set(address, observation.frame + 300);
  rollout.stalledTargetEvents.push({
    frame: observation.frame,
    address,
    type: target.type,
    variant: target.variant,
    hp,
  });
  prior.lastProgressFrame = observation.frame;
}

function greedAggressiveChoice(observation, target) {
  const p = observation.player.position;
  const dx = target.position.x - p.x;
  const dy = target.position.y - p.y;
  const horizontalMiss = Math.abs(dy);
  const verticalMiss = Math.abs(dx);
  const fire = horizontalMiss <= verticalMiss
    ? (dx < 0 ? 'left' : 'right')
    : (dy < 0 ? 'up' : 'down');
  return {
    target, dx, dy, distance: Math.hypot(dx, dy), fire,
    miss: Math.min(horizontalMiss, verticalMiss),
  };
}

function greedAggressiveAction(observation, choice) {
  let moveX = 0, moveY = 0;
  if (choice.distance > 210) {
    moveX = Math.abs(choice.dx) > 12 ? Math.sign(choice.dx) : 0;
    moveY = Math.abs(choice.dy) > 12 ? Math.sign(choice.dy) : 0;
  } else if (choice.miss > 8) {
    if (choice.fire === 'left' || choice.fire === 'right') {
      moveY = Math.abs(choice.dy) > 8 ? Math.sign(choice.dy) : 0;
    } else {
      moveX = Math.abs(choice.dx) > 8 ? Math.sign(choice.dx) : 0;
    }
  }
  rollout.aggressiveClearFrames = (rollout.aggressiveClearFrames ?? 0) + 1;
  rollout.lastDecision = {
    move: 0,
    fire: 0,
    target: choice.target.address,
    valid: true,
    safe: true,
    stop: false,
    reason: 'Greed stalled target recovery',
  };
  return { moveX, moveY, fire: choice.fire };
}

function greedAggressiveClearAction(observation) {
  if (config.greedAggressiveClear !== true || !observation.player || !observation.room) return null;
  if (observation.room.type !== 1 || observation.room.variant !== 1018 ||
      (observation.game?.difficulty | 0) < 2) return null;
  const targets = observation.entities.filter(entity =>
    isAlive(entity) && isEnemy(entity) && !entity.friendly && entity.targetable &&
    !entity.invulnerable
  );
  const hpSum = targets.reduce((sum, entity) =>
    sum + (Number.isFinite(entity.hp) ? entity.hp : 0), 0);

  let locked = targets.find(target => target.address === rollout.greedAggressiveTarget);
  if (locked && observation.frame - rollout.greedAggressiveLockFrame > 1200) {
    rollout.stalledTargetEvents.push({
      frame: observation.frame,
      address: locked.address,
      type: locked.type,
      variant: locked.variant,
      hp: locked.hp,
      reason: 'locked recovery timeout',
    });
    locked = null;
  }
  if (!locked) {
    rollout.greedAggressiveTarget = null;
    rollout.greedAggressiveLockFrame = 0;
  }

  if (locked) {
    rollout.greedAggressiveHp = hpSum;
    return greedAggressiveAction(observation, greedAggressiveChoice(observation, locked));
  }

  if (rollout.greedAggressiveHp == null || hpSum < rollout.greedAggressiveHp - 2) {
    rollout.greedAggressiveHp = hpSum;
    rollout.greedAggressiveProgressFrame = observation.frame;
    return null;
  }
  if (!targets.length || observation.frame - rollout.greedAggressiveProgressFrame < 600) return null;

  const scored = targets.map(target => {
    const choice = greedAggressiveChoice(observation, target);
    choice.score = choice.miss * 4 + choice.distance * 0.15 -
      (target.type === 30 && target.variant === 2 ? 80 : 0);
    return choice;
  }).sort((a, b) => a.score - b.score);
  const choice = scored[0];
  rollout.greedAggressiveTarget = choice.target.address;
  rollout.greedAggressiveLockFrame = observation.frame;
  return greedAggressiveAction(observation, choice);
}

function policyAction(observation) {
  if (rollout.recordGeometryTimeline) {
    trackGeometryFrame(toPlannerFrame(module, observation, rollout.attack));
  }
  if(rollout.prefixActions && rollout.frames < rollout.prefixActions.length) {
    queueAction(rollout.prefixActions[rollout.frames]);
    return;
  }
  if(rollout.actions) {queueAction(rollout.actions[rollout.frames]);return;}
  const assistAction = greedWaveAssistAction(observation);
  if (assistAction) {
    rollout.waveAssistFrames = (rollout.waveAssistFrames ?? 0) + 1;
    rollout.lastDecision = {
      move: 0,
      fire: 0,
      target: 0,
      valid: true,
      safe: true,
      stop: false,
      reason: 'Greed mode wave switch',
    };
    queueAction(assistAction);
    return;
  }
  const escapeAction = greedButtonEscapeAction(observation);
  if (escapeAction) {
    queueAction(escapeAction);
    return;
  }
  const aggressiveAction = greedAggressiveClearAction(observation);
  if (aggressiveAction) {
    queueAction(aggressiveAction);
    return;
  }
  updateGreedTargetProgress(observation);
  let result;
  let policyObservation = config.ignoreRoomClear === true && observation.cleared
    ? { ...observation, cleared: false }
    : observation;
  if (config.greedDropStalledTargets === true && rollout.suppressedTargets.size) {
    for (const [address, until] of rollout.suppressedTargets) {
      if (until <= observation.frame) rollout.suppressedTargets.delete(address);
    }
    if (rollout.suppressedTargets.size) {
      policyObservation = {
        ...policyObservation,
        entities: policyObservation.entities.map(entity => {
          if (!rollout.suppressedTargets.has(entity.address) || !isAlive(entity) || entity.friendly) {
            return entity;
          }
          return {
            ...entity,
            flags: {
              ...entity.flags,
              low: entity.flags.low | (1 << 4),
            },
          };
        }),
      };
    }
  }
  if (config.greedSackPriority === true && observation.room?.type === 1 &&
      observation.room.variant === 1018 && observation.game?.difficulty >= 2) {
    const sacks = observation.entities.filter(entity =>
      isAlive(entity) && entity.type === 30 && entity.variant === 2 && !entity.friendly
    );
    if (sacks.length) {
      policyObservation = {
        ...policyObservation,
        entities: policyObservation.entities.map(entity => {
          if (!isAlive(entity) || !isEnemy(entity) ||
              (entity.type === 30 && entity.variant === 2) || entity.friendly) {
            return entity;
          }
          return {
            ...entity,
            flags: {
              ...entity.flags,
              low: entity.flags.low | (1 << 4),
            },
          };
        }),
      };
    }
  }
  try { result = policy.decide(module, policyObservation, rollout.attack); }
  catch(error) {
    writeFileSync(join(config.run,'policy-failure-observation.json'),JSON.stringify(observation));
    send({event:'fatal',error:`policy failed at logic frame ${observation.frame}: ${error.stack??error.message}`});
    throw error;
  }
  if(rollout.lastPolicyMove!==null&&rollout.lastPolicyMove!==result.decision.move)rollout.moveSwitches++;
  rollout.lastPolicyMove=result.decision.move;
  rollout.lastDecision = result.decision;
  if (rollout.recordDecisionHistory) {
    rollout.decisionHistory.push({
      frame:result.frame.clock,
      player:result.frame.player.body.pos,
      velocity:result.frame.player.body.vel,
      playerStats:{
        tearSpeed:result.frame.player.tear_speed,
        tearRange:result.frame.player.tear_range,
        maxSpeed:result.frame.player.max_speed,
        fireDelay:result.frame.player.fire_delay,
      },
      decision:decisionSummary(result.decision),
      hazards:rollout.lastFrameHazards,
    });
    if (rollout.decisionHistory.length>rollout.decisionHistoryLimit) rollout.decisionHistory.shift();
  }
  if (!!result.decision.navigation_boost !== rollout.lastNavigationBoost) {
    const target = result.frame.entities.find(entity => entity.address === result.decision.target);
    rollout.lastNavigationBoost = !!result.decision.navigation_boost;
    rollout.decisionTransitions.push({
      frame:result.frame.clock,
      navigation_boost:rollout.lastNavigationBoost,
      target:result.decision.target,
      targetHp:target?.hp ?? null,
      targetType:target?.type ?? null,
      reason:result.decision.reason,
    });
  }
  trackFrame(result.frame);
  for (const kind of result.decision.threat_kinds ?? []) {
    rollout.decisionThreatCounts.set(kind,(rollout.decisionThreatCounts.get(kind)??0)+1);
  }
  if ((result.decision.stop || !result.decision.valid) &&
      (rollout.stopEvents.length===0 || result.frame.clock-rollout.lastStopFrame>=300)) {
    rollout.lastStopFrame=result.frame.clock;
    rollout.stopEvents.push({
      frame:result.frame.clock,
      player:result.frame.player.body.pos,
      velocity:result.frame.player.body.vel,
      decision:decisionSummary(result.decision),
      localGrid:localGrid(result.frame),
    });
  }
  if (!result.decision.safe) {
    rollout.unsafeFrames++;
    if (rollout.unsafeEvents.length<64 &&
        result.frame.clock-rollout.lastUnsafeFrame>=15) {
      rollout.lastUnsafeFrame=result.frame.clock;
      rollout.unsafeEvents.push({
        frame:result.frame.clock,
        player:result.frame.player.body.pos,
        velocity:result.frame.player.body.vel,
        decision:decisionSummary(result.decision),
        path:result.decision.path.slice(0,20),
        hazards:rollout.lastFrameHazards,
        localGrid:localGrid(result.frame),
      });
    }
  }
  queueAction(result.action);
}

export function attachEngine(value) {
  module = value;
  const setupPoll = module.isaacInputPoll ?? (() => 0);
  module.isaacInputPoll = (presented, out) => {
    if (probe?.console) return probe.console.poll(presented, out, module.HEAP32);
    if (setupConsole) {
      const result = setupConsole.poll(presented, out, module.HEAP32);
      if (warmReset && config.setupTrace && warmReset.inputPolls++ < 24) {
        console.log(`[training][warm-reset] input poll presented=${presented} event=${result} done=${setupConsole.done()}`);
      }
      return result;
    }
    if (!active) return setupPoll(presented, out);
    const event = events.shift();
    if (!event) return 0;
    module.HEAP32.set(event, out >> 2);
    return 1;
  };
  module.isaacTrainingBoundary = () => {
    const observation = readObservation(module);
    if (config.externalValidation === true && config.externalBoundaryLog === true) {
      appendFileSync(join(config.run,'validation-boundaries.jsonl'),JSON.stringify({frame:observation.frame,state:observation.state,room:observation.room})+'\n');
    }
    const game = observation.game?.pointer;
    const heap = module.HEAPU8;
    const consoleState = game ? new DataView(heap.buffer).getUint32(game + 0x68d78, true) : null;
    if (!forcedDifficultyApplied &&
        Number.isInteger(config.forcedDifficulty) &&
        game &&
        observation.state === 'ready') {
      new DataView(heap.buffer).setInt32(
        game + TRAINING_OBSERVER_OFFSETS.game.difficulty,
        config.forcedDifficulty,
        true,
      );
      forcedDifficultyApplied = true;
      console.log(`[training] forced game difficulty=${config.forcedDifficulty} at logic frame ${observation.frame}`);
    }
    const targetSeen = allTargetsSeen(observation);
    const roomReady = observation.ready && observation.player?.controls &&
      (!observation.cleared || hasSpawnRequest()) &&
      consoleState === 0 && observation.game.roomIndex < 0;
    if (warmReset) {
      warmReset.polls++;
      frame = observation.frame;
      if (config.setupTrace && (warmReset.polls <= 5 || warmReset.polls % 120 === 0)) {
        console.log(`[training][warm-reset] boundary frame=${observation.frame} state=${observation.state} console=${consoleState} done=${setupConsole?.done() ?? null}`);
      }
      if (setupConsole && (consoleState !== 0 || !setupConsole.done())) return;
      if (setupConsole) {
        setupConsole = null;
        warmReset.seedState = seedLogState();
      }
      const targetPresent = observation.entities.some(entity =>
        entity.type === warmReset.targetNPC && isAlive(entity)
      );
      const roomMatches = observation.room?.type === warmReset.roomType &&
        observation.room?.variant === warmReset.roomVariant;
      const seedConfirmed = warmReset.seedState?.count > warmReset.baselineSeedCount &&
        warmReset.seedState?.seed === warmReset.seed;
      const resetReady = observation.ready === true && observation.player?.controls === true &&
        consoleState === 0 && observation.game?.transition !== true && roomMatches && targetPresent && seedConfirmed;
      if (resetReady) {
        if (observation.frame !== warmReset.lastStableFrame) warmReset.stableFrames++;
        warmReset.lastStableFrame = observation.frame;
      } else {
        warmReset.stableFrames = 0;
        warmReset.lastStableFrame = null;
      }
      if (warmReset.stableFrames >= 2) {
        const completed = warmReset;
        warmReset = null;
        setupConsole = null;
        setupConsoleTick = 0;
        setupTargetSeen = true;
        setupObservation = observation;
        setupStartedFrame = -1;
        setupIsolationPasses = 0;
        active = true;
        module.isaacTrainingNoRender = config.noRender === true;
        frame = observation.frame;
        steps = 0;
        terminal = observation.dead === true ||
          (observation.cleared === true && config.ignoreRoomClear !== true);
        previousAction = null;
        events.length = 0;
        send({ id: completed.id, result: { observation, resetMode: 'warm', seed: completed.seed, room: completed.room } });
        return new Promise(resolve => { resume = resolve; });
      }
      if (warmReset.polls > warmReset.maxPolls) {
        const failed = warmReset;
        warmReset = null;
        setupConsole = null;
        active = true;
        module.isaacTrainingNoRender = config.noRender === true;
        frame = observation.frame;
        events.length = 0;
        previousAction = null;
        send({ id: failed.id, error: `warm reset timed out for ${failed.room} / ${failed.seed} at frame ${observation.frame}` });
        return new Promise(resolve => { resume = resolve; });
      }
      return;
    }
    if (!active && config.setupTrace && observation.frame - lastSetupTraceFrame >= 60) {
      lastSetupTraceFrame = observation.frame;
      const targetType = targetSpec().type;
      const targetState = observation.entities
        .filter(entity => entity.type === targetType)
        .map(entity => `${entity.type}.${entity.variant}.${entity.subtype}:active=${entity.active},exists=${entity.exists},dead=${entity.dead}`)
        .join(',');
      console.log(`[training] setup frame=${observation.frame} state=${observation.state} roomIndex=${observation.game.roomIndex} controls=${observation.player?.controls} cleared=${observation.cleared} console=${consoleState} consoleDone=${setupConsole?.done() ?? null} setupStarted=${setupStartedFrame} context=${config.contextMode===true} targetSeen=${targetSeen} target=${targetState} identities=${observation.entities.filter(isEnemy).map(entity => `${entity.type}.${entity.variant}.${entity.subtype}`).join(',')}`);
    }
    if (!setupConsole && !hasSpawnRequest() && targetSeen) {
      setupTargetSeen = true;
      setupObservation ??= observation;
    }
    if (setupConsole) frame = observation.frame;
    if (handleProbe(observation,roomReady,consoleState)) return;
    if (setupConsole) {
      if (consoleState !== 0) return;
      if (!setupConsole.done()) return;
      if (config.contextMode &&
          observation.frame - setupStartedFrame >= (config.contextWaitFrames ?? 30)) {
        const setup = setupConsole.report();
        setupConsole = null;
        active = true;
        module.isaacTrainingNoRender = config.noRender === true;
        frame = observation.frame;
        send({ event: 'ready', observation, setup });
        return;
      }
      if (targetSeen) {
        setupTargetSeen = true;
        setupObservation ??= observation;
      }
      const allowedTypes=new Set(config.isolationTypes??[]);
      const unexpected=observation.entities.filter(entity=>
        isAlive(entity)&&isEnemy(entity)&&!allowedTypes.has(entity.type)
      );
      const isolationCommand=config.isolationCleanupCommand;
      if(setupTargetSeen&&unexpected.length&&isolationCommand&&
         setupIsolationPasses<(config.isolationCleanupPasses??3)) {
        setupIsolationPasses++;
        frame=observation.frame;
        startSetupConsole([isolationCommand],'isolation');
        return;
      }
      if (setupTargetSeen && consoleState === 0) {
        const setup = {
          ...setupConsole.report(),
          target: setupObservation?.entities.find(entity =>
            targetSpecs().some(target => isTarget(entity,target))
          ) ?? null,
        };
        setupConsole = null;
        active = true;
        module.isaacTrainingNoRender = config.noRender === true;
        frame = observation.frame;
        send({ event: 'ready', observation, setup });
      } else if ((setupConsole.done() &&
                  observation.frame - setupStartedFrame > (config.postSpawnWaitFrames ?? 180)) ||
                 observation.frame - setupStartedFrame > (config.setupMaxFrames ?? 2400)) {
        const setup = setupConsole.report();
        const error = `spawn ${JSON.stringify(config.spawnNPCs ?? config.spawnNPC)} did not appear after ${observation.frame-setupStartedFrame} frames; setup=${JSON.stringify(setup)}`;
        send({ event: 'fatal', error });
        throw new Error(error);
      }
      return;
    }
    if (!active) {
      if (config.contextMode &&
          (config.contextNonReady === true || observation.state === 'ready') &&
          (consoleState === 0 || !observation.game?.pointer) &&
          observation.frame >= (config.contextActivateFrame ?? 60) &&
          observation.game?.managerPointer &&
          (config.contextNonReady === true || observation.game?.pointer)) {
        active = true;
        module.isaacTrainingNoRender = config.noRender === true;
        frame = observation.frame;
        send({ event: 'ready', observation, setup: null });
        return new Promise(resolve => { resume = resolve; });
      }
      if (!roomReady) return;
      if (config.catalogMode) {
        active = true;
        module.isaacTrainingNoRender = config.noRender === true;
        frame = observation.frame;
        send({ event: 'ready', observation, setup: null });
        return new Promise(resolve => { resume = resolve; });
      }
      if (hasSpawnRequest() && setupStartedFrame < 0) {
        frame = observation.frame;
        const requestedSpawns = config.spawnNPCs?.length>0?config.spawnNPCs:[config.spawnNPC];
        const spawns = requestedSpawns.map(spawn =>
          typeof spawn === 'string' ? spawn : spawnIdentityCommand(spawn)
        );
        setupIsolationPasses = 0;
        startSetupConsole([
          ...(config.preSpawnCommands ?? []),
          ...spawns,
        ], 'spawn');
        return;
      }
      if (!setupTargetSeen) return;
      active = true;
      module.isaacTrainingNoRender = config.noRender === true;
      frame = observation.frame;
      send({ event: 'ready', observation, setup: null });
    } else {
      if (observation.frame === frame) return;
      if (observation.frame !== frame + 1) {
        send({ event: 'fatal', error: `logic frame jumped ${frame} -> ${observation.frame}` });
        throw new Error('logic frame boundary skipped');
      }
      frame = observation.frame;
      steps++;
      terminal = observation.dead === true ||
        (observation.cleared === true && config.ignoreRoomClear !== true);
      if (pendingStep) {
        send({ id: pendingStep.id, result: { observation, terminated: terminal, truncated: steps >= config.maxSteps, steps } });
        pendingStep = null;
      }
      if (rollout) {
        const hp = observation.player?.health;
        const cooldown = observation.player?.damageCooldown;
        const cooldownHit = isCooldownHit(rollout.lastDamageCooldown,cooldown,config.damageCalibration);
        rollout.frames++;
        if(rollout.hashes)rollout.hashes.push(stateHash(observation));
        const damage = Number.isFinite(hp) ? Math.max(0, rollout.lastHealth - hp) : 0;
        const hit = damage>0 || cooldownHit;
        let hitDetails = null;
        if (hit || (damage>0 && rollout.damageEvents.length<64)) {
          rollout.hitCount++;
          if (rollout.hitEvents.length<64 || (damage>0 && rollout.damageEvents.length<64)) {
            try {
              hitDetails = damageDetails(observation, damage, rollout.lastHealth, {
                healthDamage:damage,
                healthUnchanged:damage===0,
              });
            } catch (error) {
              hitDetails = {frame:observation.frame,amount:damage,
                priorHealth:rollout.lastHealth,health:hp,error:error.stack??error.message};
            }
          }
          if (rollout.hitEvents.length<64) rollout.hitEvents.push(hitDetails);
        }
        if (damage>0) {
          rollout.damageTaken += damage;
          if (rollout.damageEvents.length<64) rollout.damageEvents.push(hitDetails);
        }
        if (Number.isFinite(hp)) rollout.lastHealth = hp;
        rollout.lastDamageCooldown = cooldown;
        rollout.peakProjectiles = Math.max(rollout.peakProjectiles, observation.entities.filter(e => e.type===9 && e.dead===false).length);
        rollout.peakTears = Math.max(rollout.peakTears, observation.entities.filter(e => e.type===2 && e.dead===false).length);
        if (terminal || rollout.frames >= rollout.limit || steps >= config.maxSteps) {
          flushGeometryTimeline();
          const summary = { frames:rollout.frames, damageTaken:rollout.damageTaken, health:hp,
            cleared:observation.cleared, dead:observation.dead, peakTears:rollout.peakTears,
            peakProjectiles:rollout.peakProjectiles, elapsedMs:performance.now()-rollout.started,
            parameters:rollout.parameters, lastDecision:rollout.lastDecision, maxPolicyInputBytes:policy.maxInputBytes,
            hitCount:rollout.hitCount, hitEvents:rollout.hitEvents, damageEvents:rollout.damageEvents,
            moveSwitches:rollout.moveSwitches,
            waveAssistFrames:rollout.waveAssistFrames,
            greedRegularWaveStarted:rollout.waveAssist.regularStarted,
            greedBossWaveStarted:rollout.waveAssist.bossStarted,
            greedButtonEscapeFrames:rollout.greedButtonEscapeFrames ?? 0,
            stalledTargetEvents:rollout.stalledTargetEvents,
            aggressiveClearFrames:rollout.aggressiveClearFrames,
            unsafeFrames:rollout.unsafeFrames, unsafeEvents:rollout.unsafeEvents,
            stopEvents:rollout.stopEvents,
            decisionTransitions:rollout.decisionTransitions,
            decisionHistory:rollout.recordDecisionHistory ? rollout.decisionHistory.slice(-rollout.decisionHistoryLimit) : [],
            entityTransitions:rollout.entityTransitions,
            enemyHpTransitions:rollout.enemyHpTransitions,
            geometryTimeline:rollout.recordGeometryTimeline?{
              file:join(config.run,'geometry-timeline.jsonl'),
              bytes:rollout.geometryBytes,
              frames:rollout.geometryFrames,
              actor_samples:rollout.geometryActorSamples,
              hazard_samples:rollout.geometryHazardSamples,
              hazard_kinds:Object.fromEntries([...rollout.geometryHazardKinds].sort(([a],[b])=>a.localeCompare(b))),
              source_identities:[...rollout.geometrySourceIdentities].sort(),
            }:null,
            enemyTypes:[...rollout.enemyTypes.values()].sort((a,b)=>a.firstFrame-b.firstFrame || a.type-b.type),
            decisionThreatCounts:Object.fromEntries([...rollout.decisionThreatCounts].sort(([a],[b])=>a.localeCompare(b))),
            dangerFrameCounts:Object.fromEntries([...rollout.dangerFrameCounts].sort(([a],[b])=>a.localeCompare(b))) };
          summary.logicFramesPerSecond = summary.frames / (summary.elapsedMs / 1000);
          summary.moveSwitchesPer1000Frames = summary.frames ? 1000 * summary.moveSwitches / summary.frames : null;
          writeFileSync(join(config.run, 'rollout.json'), JSON.stringify(summary,null,2));
          send({ id:rollout.id, result:{observation,summary,hashes:rollout.hashes,terminated:terminal,truncated:!terminal} });
          rollout=null;
        } else { policyAction(observation); return; }
      }
    }
    return new Promise(resolve => { resume = resolve; });
  };
}

process.on('message', message => {
  try {
    if (message.command === 'reset') {
      if (!active || !resume || pendingStep || rollout || setupConsole || probe || warmReset) {
        throw new Error('warm reset requires an idle game-frame boundary');
      }
      const options = message.action ?? {};
      const seed = String(options.seed ?? '').trim().toUpperCase().replace(/\s+/g, ' ');
      const room = String(options.room ?? '');
      const targetNPC = Number(options.targetNPC);
      if (!/^[A-Z0-9]{4} [A-Z0-9]{4}$/.test(seed)) throw new Error('invalid native seed');
      const bossRoom = /^s\.boss\.\d+$/.test(room);
      const gaperRoom = room === 'd.841' && targetNPC === 10;
      if (!bossRoom && !gaperRoom) throw new Error('warm reset room must be a supported scenario');
      if (!Number.isInteger(targetNPC) || targetNPC < 10 || targetNPC >= 1000) throw new Error('invalid warm reset target NPC');
      const roomVariant = Number(bossRoom ? room.slice('s.boss.'.length) : room.slice('d.'.length));
      const seedState = seedLogState();
      trainingSnapshots.clear();
      trainingForks.clear();
      config.seed = seed;
      config.room = room;
      config.targetNPC = targetNPC;
      steps = 0;
      terminal = false;
      events.length = 0;
      queueAction({ moveX: 0, moveY: 0, fire: 'none' });
      // Console typing advances on the original presentation-frame counter.
      // Restore presents only while native reset commands are being entered.
      module.isaacTrainingNoRender = false;
      warmReset = {
        id: message.id,
        seed,
        room,
        roomType: gaperRoom ? 1 : 5,
        roomVariant,
        targetNPC,
        baselineSeedCount: seedState.count,
        seedState: null,
        stableFrames: 0,
        lastStableFrame: null,
        polls: 0,
        inputPolls: 0,
        maxPolls: Number(options.maxPolls ?? 6000),
      };
      startSetupConsole([
        `seed ${seed}`,
        `stage ${Number.isInteger(options.stage) ? options.stage : 1}`,
        `goto ${room}`,
      ], 'warm-reset');
      const go = resume; resume = null; go();
    } else if (message.command === 'snapshot') {
      send({ id: message.id, result: snapshotTrainingState(message.action?.reuseId ?? null) });
    } else if (message.command === 'restore') {
      send({ id: message.id, result: restoreTrainingState(message.action?.snapshotId ?? message.action?.id) });
    } else if (message.command === 'fork') {
      if (!trainingBoundaryIsIdle()) throw new Error('fork requires an idle, ready game-frame boundary');
      send({ id: message.id, result: forkTrainingState(message.action?.snapshotId ?? message.action?.id, message.action?.label ?? '') });
    } else if (message.command === 'dropSnapshot') {
      send({ id: message.id, result: dropTrainingState(message.action?.id) });
    } else if (message.command === 'snapshot_export') {
      send({ id: message.id, result: exportTrainingSnapshot(message.action?.id, message.action?.path) });
    } else if (message.command === 'snapshot_import') {
      send({ id: message.id, result: importTrainingSnapshot(message.action?.path) });
    } else if (message.command === 'externalTruth' && config.externalValidation === true) {
      if (!active || !resume || pendingStep) throw new Error('engine is not stopped at a logic boundary');
      import(pathToFileURL(join(config.research, 'external-validation', 'truth.mjs')).href).then(({captureTruth}) => {
        send({id:message.id,result:captureTruth(module,readObservation(module))});
      }).catch(error => send({id:message.id,error:error.stack??error.message}));
    } else if (message.command === 'peek') {
      if (!active || !resume) throw new Error('engine is not waiting at a logic boundary');
      send({ id: message.id, result: readObservation(module) });
    } else if (message.command === 'step') {
      if (!active || !resume || pendingStep) throw new Error('one step is already running');
      if (terminal || steps >= config.maxSteps) throw new Error('episode ended; reset before stepping');
      queueAction(message.action);
      pendingStep = message;
      const go = resume; resume = null; go();
    } else if (message.command === 'key') {
      if (!active || !resume || pendingStep) throw new Error('engine is not waiting for a key');
      const name = message.action?.name;
      const key = KEYS[name];
      if (!key) throw new Error(`unknown key ${name}`);
      const down = message.action?.down !== false;
      const [vk, scan, extended] = key;
      events.push([1, vk, scan | (extended << 8), down ? 1 : 0]);
      pendingStep = message;
      const go = resume; resume = null; go();
    } else if(message.command === 'rollout') {
      if(!active||!resume||pendingStep||rollout)throw new Error('engine is not waiting for a rollout');
      if(terminal||steps>=config.maxSteps)throw new Error('episode ended; reset first');
      const options=message.action??{}, observation=readObservation(module);
      const parameters={dt:1/30,contact_margin:8,...(options.parameters??{})};
      policy.configure(parameters);
      rollout={id:message.id,limit:Math.min(
          options.actions?.length ?? options.steps ?? 1500,
          config.maxSteps-steps),frames:0,
        started:performance.now(),lastHealth:observation.player.health,
        lastDamageCooldown:observation.player.damageCooldown,damageTaken:0,hitCount:0,hitEvents:[],
        peakProjectiles:0,peakTears:0,parameters,attack:options.attack!==false,
        actions:options.actions,prefixActions:options.prefixActions,
        waveAssistFrames:0,
        greedButtonEscapeFrames:0,
        waveAssist:{phase:'idle',phaseFrames:0,cooldown:0,regularStarted:false,bossStarted:false},
        targetProgress:new Map(),suppressedTargets:new Map(),stalledTargetEvents:[],
        greedAggressiveHp:null,greedAggressiveProgressFrame:0,aggressiveClearFrames:0,
        greedAggressiveTarget:null,greedAggressiveLockFrame:0,
        hashes:options.recordHashes?[stateHash(observation)]:null,damageEvents:[],stopEvents:[],lastStopFrame:-300,
        unsafeEvents:[],unsafeFrames:0,lastUnsafeFrame:-100,
        enemyTypes:new Map(),dangerFrameKinds:new Map(),dangerFrameCounts:new Map(),
        decisionThreatCounts:new Map(),
        moveSwitches:0,lastPolicyMove:null,decisionTransitions:[],lastNavigationBoost:null,
        recordDecisionHistory:options.recordDecisionHistory===true,decisionHistory:[],
        decisionHistoryLimit:Math.max(1,Number(options.decisionHistoryLimit??120)),
        recordEntityTypes:new Set(options.recordEntityTimeline??[]),entityTransitions:[],
        enemyHpTransitions:[],lastEntityTransition:new Map(),lastFrameHazards:[],
        recordGeometryTimeline:options.recordGeometryTimeline===true,
        geometryBuffer:[],geometryBytes:0,geometryFrames:0,geometryActorSamples:0,
        geometryHazardSamples:0,geometryHazardKinds:new Map(),geometrySourceIdentities:new Set(),
        lastGeometryFrame:-1};
      if(rollout.recordGeometryTimeline)
        writeFileSync(join(config.run,'geometry-timeline.jsonl'),'');
      if(rollout.recordGeometryTimeline)
        trackGeometryFrame(toPlannerFrame(module, observation, rollout.attack));
      policyAction(observation);
      const go=resume;resume=null;go();
    } else if(message.command === 'probe') {
      if(!active||!resume||rollout||setupConsole)throw new Error('engine is not waiting for a catalog probe');
      startProbe(message);
      const go=resume;resume=null;go();
    } else throw new Error(`unknown command ${message.command}`);
  } catch (error) { send({ id: message.id, error: error.message }); }
});

process.env.ISAAC_INSTANCE_DIR = config.instance;
if (config.persistSaves === true) process.env.ISAAC_SAVE_DIR = join(config.run, 'saves');
else delete process.env.ISAAC_SAVE_DIR;
process.env.ISAAC_EPOCH = '1793118193';
process.env.ISAAC_MAX_FRAMES = '2000000';
process.env.ISAAC_DRIVE = config.drive ?? 'explore';
if(config.inputTimeline !== undefined) process.env.ISAAC_INPUT = config.inputTimeline;
const startupCommands=config.startupCommands ? [...config.startupCommands] : [];
if(!config.startupCommands) {
  for(const command of config.preRoomCommands??[])startupCommands.push(command);
  if(config.stage!==undefined)startupCommands.push(`stage ${config.stage}`);
  startupCommands.push(`seed ${config.seed}`);
  for(const flag of config.debugFlags??[])startupCommands.push(`debug ${flag}`);
  startupCommands.push(`goto ${config.room}`);
}
process.env.ISAAC_CONSOLE = startupCommands.join(';');
process.env.ISAAC_CONSOLE_MODE = 'typed';
process.env.ISAAC_CONSOLE_DELAY = '90';
if(config.consoleAt !== undefined) process.env.ISAAC_CONSOLE_AT = String(config.consoleAt);
process.env.ISAAC_OBSERVER_MODULE = fileURLToPath(import.meta.url);
process.env.ISAAC_V8_FLAGS = '';
process.env.ISAAC_V8_FLAGS_APPLIED = '1';
process.env.ISAAC_YIELD = '0';
process.argv = [process.execPath, driver, join(config.research, 'runtime-inputs/host/isaac.segs.bin'), 'main'];
writeFileSync(join(config.run, 'configuration.json'), JSON.stringify(config, null, 2));
import(pathToFileURL(driver).href).then(() => {
  send({ event: 'fatal', error: 'engine returned before the training session closed' });
}).catch(error => {
  send({ event: 'fatal', error: error.stack ?? error.message });
});
