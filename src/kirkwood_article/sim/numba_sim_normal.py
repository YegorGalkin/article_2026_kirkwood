# ruff: noqa
# Vendored/adapted from https://github.com/YegorGalkin/SBDPP_sim/blob/master/SSA/numba_sim_normal.py
# Used as the high-throughput SSA backend for long Kirkwood scaling simulations.
"""
Spatial Stochastic Simulator with Normal (Gaussian) Kernels.
Supports 1D, 2D, 3D with periodic or killing boundaries.
Birth dispersal: np.random.normal, no max distance.
Death kernel: normalized Gaussian K(r) = (2πσ²)^(-d/2) * exp(-r²/(2σ²))
"""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from numba import njit, types
from numba.experimental import jitclass

@njit(cache=True, inline='always')
def _distance_nd(pos1: NDArray, pos2: NDArray, area: NDArray, ndim: int, periodic: bool) -> float:
    """Calculate distance with optional periodic boundaries."""
    dsq = 0.0
    for d in range(ndim):
        diff = abs(pos1[d] - pos2[d])
        if periodic and area[d] > 0.0:
            wrap = area[d] - diff
            if wrap < diff:
                diff = wrap
        dsq += diff * diff
    return math.sqrt(dsq)

@njit(cache=True, inline='always')
def _wrap_position(pos: NDArray, area: NDArray, ndim: int, periodic: bool) -> bool:
    """Wrap/validate position. Returns True if valid."""
    if periodic:
        for d in range(ndim):
            if area[d] > 0.0:
                pos[d] = pos[d] - area[d] * math.floor(pos[d] / area[d])
                if pos[d] < 0.0:
                    pos[d] += area[d]
        return True
    else:
        for d in range(ndim):
            if pos[d] < 0.0 or pos[d] > area[d]:
                return False
        return True

@njit(cache=True, inline='always')
def _gaussian_kernel(r: float, sigma: float, ndim: int) -> float:
    """Normalized Gaussian kernel: (2πσ²)^(-d/2) * exp(-r²/(2σ²))"""
    if sigma <= 0.0:
        return 0.0
    var = sigma * sigma
    norm = math.pow(2.0 * math.pi * var, -0.5 * ndim)
    return norm * math.exp(-r * r / (2.0 * var))

@njit(cache=True, inline='always')
def _sample_weighted(values: NDArray[np.float64]) -> int:
    """Sample an index from weighted distribution."""
    total = 0.0
    for i in range(values.shape[0]):
        total += values[i]
    if total <= 0.0:
        return -1
    r = np.random.random() * total
    cumsum = 0.0
    for i in range(values.shape[0]):
        cumsum += values[i]
        if r <= cumsum:
            return i
    return values.shape[0] - 1

ssa_normal_spec = [
    ("ndim", types.int32),
    ("species_count", types.int32),
    ("area_size", types.Array(types.float64, 1, "C")),
    ("cell_counts", types.Array(types.int32, 1, "C")),
    ("cell_size", types.Array(types.float64, 1, "C")),
    ("periodic", types.boolean),
    ("b", types.Array(types.float64, 1, "C")),
    ("d", types.Array(types.float64, 1, "C")),
    ("dd", types.Array(types.float64, 2, "C")),
    ("birth_std", types.Array(types.float64, 1, "C")),
    ("death_std", types.Array(types.float64, 2, "C")),
    ("cutoff", types.Array(types.float64, 2, "C")),
    ("cull_range", types.Array(types.int32, 2, "C")),
    ("capacity", types.int64),
    ("population", types.int64),
    ("positions", types.Array(types.float64, 2, "C")),
    ("species_id", types.Array(types.int32, 1, "C")),
    ("death_rate", types.Array(types.float64, 1, "C")),
    ("total_cells", types.int64),
    ("cell_capacity", types.int32),
    ("particle_cell", types.Array(types.int64, 1, "C")),
    ("particle_slot", types.Array(types.int32, 1, "C")),
    ("cell_particles", types.Array(types.int64, 2, "C")),
    ("cell_particle_count", types.Array(types.int32, 1, "C")),
    ("cell_species_count", types.Array(types.int32, 2, "C")),
    ("cell_birth_rate", types.Array(types.float64, 1, "C")),
    ("cell_death_rate", types.Array(types.float64, 1, "C")),
    ("total_birth_rate", types.float64),
    ("total_death_rate", types.float64),
    ("time", types.float64),
    ("event_count", types.int64),
]

@jitclass(ssa_normal_spec)
class SSANormalState:
    """SSA with Normal kernels for birth dispersal and death interactions."""
    
    def __init__(self, ndim: np.int32, species_count: np.int32, area_size: NDArray,
                 cell_counts: NDArray, periodic: bool, capacity: np.int64,
                 cell_capacity: np.int32, seed: np.int64,
                 b: NDArray, d: NDArray, dd: NDArray, birth_std: NDArray,
                 death_std: NDArray, death_cull_sigmas: float):
        self.ndim = ndim
        self.species_count = species_count
        self.area_size = area_size
        self.cell_counts = cell_counts
        self.periodic = periodic
        self.b, self.d, self.dd = b, d, dd
        self.birth_std = birth_std
        self.death_std = death_std
        
        self.cell_size = np.empty(ndim, dtype=np.float64)
        for dim in range(ndim):
            self.cell_size[dim] = area_size[dim] / float(cell_counts[dim])
        
        # Compute cutoffs from death_std * cull_sigmas
        self.cutoff = np.empty((species_count, species_count), dtype=np.float64)
        self.cull_range = np.zeros((species_count, species_count), dtype=np.int32)
        max_cell = 0.0
        for dim in range(ndim):
            if self.cell_size[dim] > max_cell:
                max_cell = self.cell_size[dim]
        for i in range(species_count):
            for j in range(species_count):
                self.cutoff[i, j] = death_cull_sigmas * death_std[i, j]
                if self.cutoff[i, j] > 0.0:
                    self.cull_range[i, j] = int(math.ceil(self.cutoff[i, j] / max_cell))
        
        # Particle storage
        self.capacity = capacity
        self.population = np.int64(0)
        self.positions = np.zeros((capacity, ndim), dtype=np.float64)
        self.species_id = np.full(capacity, -1, dtype=np.int32)
        self.death_rate = np.zeros(capacity, dtype=np.float64)
        
        # Cell indexing
        self.total_cells = np.int64(1)
        for dim in range(ndim):
            self.total_cells *= cell_counts[dim]
        self.cell_capacity = cell_capacity
        self.particle_cell = np.full(capacity, -1, dtype=np.int64)
        self.particle_slot = np.full(capacity, -1, dtype=np.int32)
        self.cell_particles = np.full((self.total_cells, cell_capacity), -1, dtype=np.int64)
        self.cell_particle_count = np.zeros(self.total_cells, dtype=np.int32)
        self.cell_species_count = np.zeros((self.total_cells, species_count), dtype=np.int32)
        self.cell_birth_rate = np.zeros(self.total_cells, dtype=np.float64)
        self.cell_death_rate = np.zeros(self.total_cells, dtype=np.float64)
        
        self.total_birth_rate = 0.0
        self.total_death_rate = 0.0
        self.time = 0.0
        self.event_count = np.int64(0)
        
        if seed >= 0:
            np.random.seed(int(seed))
    
    def _pos_to_cell(self, pos: NDArray) -> int:
        """Convert position to cell index."""
        if self.ndim == 1:
            ix = min(max(int(pos[0] / self.cell_size[0]), 0), self.cell_counts[0] - 1)
            return ix
        elif self.ndim == 2:
            ix = min(max(int(pos[0] / self.cell_size[0]), 0), self.cell_counts[0] - 1)
            iy = min(max(int(pos[1] / self.cell_size[1]), 0), self.cell_counts[1] - 1)
            return iy * self.cell_counts[0] + ix
        else:
            ix = min(max(int(pos[0] / self.cell_size[0]), 0), self.cell_counts[0] - 1)
            iy = min(max(int(pos[1] / self.cell_size[1]), 0), self.cell_counts[1] - 1)
            iz = min(max(int(pos[2] / self.cell_size[2]), 0), self.cell_counts[2] - 1)
            return (iz * self.cell_counts[1] + iy) * self.cell_counts[0] + ix
    
    def _double_capacity(self) -> None:
        """Double particle capacity."""
        old = self.capacity
        new = old * 2
        new_pos = np.zeros((new, self.ndim), dtype=np.float64)
        new_sp = np.full(new, -1, dtype=np.int32)
        new_dr = np.zeros(new, dtype=np.float64)
        new_pc = np.full(new, -1, dtype=np.int64)
        new_ps = np.full(new, -1, dtype=np.int32)
        for i in range(old):
            for d in range(self.ndim):
                new_pos[i, d] = self.positions[i, d]
            new_sp[i] = self.species_id[i]
            new_dr[i] = self.death_rate[i]
            new_pc[i] = self.particle_cell[i]
            new_ps[i] = self.particle_slot[i]
        self.positions = new_pos
        self.species_id = new_sp
        self.death_rate = new_dr
        self.particle_cell = new_pc
        self.particle_slot = new_ps
        self.capacity = new
    
    def _double_cell_capacity(self) -> None:
        """Double cell capacity."""
        old = self.cell_capacity
        new = old * 2
        new_cp = np.full((self.total_cells, new), -1, dtype=np.int64)
        for c in range(self.total_cells):
            for s in range(old):
                new_cp[c, s] = self.cell_particles[c, s]
        self.cell_particles = new_cp
        self.cell_capacity = new
    
    def spawn_particle(self, species: int, x: float, y: float = 0.0, z: float = 0.0) -> bool:
        """Spawn particle at position."""
        pos = np.empty(self.ndim, dtype=np.float64)
        pos[0] = x
        if self.ndim >= 2: pos[1] = y
        if self.ndim >= 3: pos[2] = z
        return self._spawn_impl(species, pos)
    
    def spawn_random(self, species: int, count: int = 1) -> int:
        """Spawn particles at random uniform positions."""
        spawned = 0
        for _ in range(count):
            pos = np.empty(self.ndim, dtype=np.float64)
            for d in range(self.ndim):
                pos[d] = np.random.uniform(0.0, self.area_size[d])
            if self._spawn_impl(species, pos):
                spawned += 1
        return spawned
    
    def _spawn_impl(self, species: int, pos: NDArray) -> bool:
        """Internal spawn implementation."""
        if self.population >= self.capacity:
            self._double_capacity()
        
        if not _wrap_position(pos, self.area_size, self.ndim, self.periodic):
            return False
        
        cell_id = self._pos_to_cell(pos)
        slot = self.cell_particle_count[cell_id]
        if slot >= self.cell_capacity:
            self._double_cell_capacity()
        
        pid = int(self.population)
        self.population += 1
        for d in range(self.ndim):
            self.positions[pid, d] = pos[d]
        self.species_id[pid] = species
        self.particle_cell[pid] = cell_id
        self.particle_slot[pid] = slot
        self.cell_particles[cell_id, slot] = pid
        self.cell_particle_count[cell_id] += 1
        self.cell_species_count[cell_id, species] += 1
        
        base_dr = self.d[species]
        self.death_rate[pid] = base_dr
        self.total_birth_rate += self.b[species]
        self.total_death_rate += base_dr
        self.cell_birth_rate[cell_id] += self.b[species]
        self.cell_death_rate[cell_id] += base_dr
        
        # Death interactions
        added = self._compute_interactions_spawn(pid, species, cell_id, pos)
        if added != 0.0:
            self.death_rate[pid] += added
            self.cell_death_rate[cell_id] += added
            self.total_death_rate += added
        
        return True
    
    def _compute_interactions_spawn(self, pid: int, species: int, cell_id: int, pos: NDArray) -> float:
        """Compute death rate changes when spawning."""
        added = 0.0
        for osp in range(self.species_count):
            cutoff_ij = self.cutoff[species, osp]
            dd_ij = self.dd[species, osp]
            cutoff_ji = self.cutoff[osp, species]
            dd_ji = self.dd[osp, species]
            if dd_ij == 0.0 and dd_ji == 0.0:
                continue
            cull = max(self.cull_range[species, osp], self.cull_range[osp, species])
            
            # Iterate neighbor cells
            for ncell, ncount in self._iter_neighbors(cell_id, cull):
                for s in range(ncount):
                    oid = self.cell_particles[ncell, s]
                    if oid == -1 or oid == pid:
                        continue
                    if self.species_id[oid] != osp:
                        continue
                    opos = self.positions[oid]
                    dist = _distance_nd(pos, opos, self.area_size, self.ndim, self.periodic)
                    
                    # Effect on others
                    if dd_ij != 0.0 and dist <= cutoff_ij:
                        sigma = self.death_std[species, osp]
                        delta = dd_ij * _gaussian_kernel(dist, sigma, self.ndim)
                        self.death_rate[oid] += delta
                        self.cell_death_rate[ncell] += delta
                        self.total_death_rate += delta
                    
                    # Effect on self
                    if dd_ji != 0.0 and dist <= cutoff_ji:
                        sigma = self.death_std[osp, species]
                        added += dd_ji * _gaussian_kernel(dist, sigma, self.ndim)
        return added
    
    def _iter_neighbors(self, cell_id: int, cull: int):
        """Yield (cell_idx, count) for neighbor cells within cull range."""
        results = []
        if self.ndim == 1:
            cx = cell_id
            for dx in range(-cull, cull + 1):
                nx = cx + dx
                if self.periodic:
                    nx = nx % self.cell_counts[0]
                elif nx < 0 or nx >= self.cell_counts[0]:
                    continue
                results.append((nx, self.cell_particle_count[nx]))
        elif self.ndim == 2:
            nx_cells = self.cell_counts[0]
            cx = cell_id % nx_cells
            cy = cell_id // nx_cells
            for dy in range(-cull, cull + 1):
                ny = cy + dy
                if self.periodic:
                    ny = ny % self.cell_counts[1]
                elif ny < 0 or ny >= self.cell_counts[1]:
                    continue
                for dx in range(-cull, cull + 1):
                    nx = cx + dx
                    if self.periodic:
                        nx = nx % self.cell_counts[0]
                    elif nx < 0 or nx >= self.cell_counts[0]:
                        continue
                    ncell = ny * nx_cells + nx
                    results.append((ncell, self.cell_particle_count[ncell]))
        else:
            nx_cells = self.cell_counts[0]
            ny_cells = self.cell_counts[1]
            cx = cell_id % nx_cells
            t = cell_id // nx_cells
            cy = t % ny_cells
            cz = t // ny_cells
            for dz in range(-cull, cull + 1):
                nz = cz + dz
                if self.periodic:
                    nz = nz % self.cell_counts[2]
                elif nz < 0 or nz >= self.cell_counts[2]:
                    continue
                for dy in range(-cull, cull + 1):
                    ny = cy + dy
                    if self.periodic:
                        ny = ny % self.cell_counts[1]
                    elif ny < 0 or ny >= self.cell_counts[1]:
                        continue
                    for dx in range(-cull, cull + 1):
                        nx = cx + dx
                        if self.periodic:
                            nx = nx % self.cell_counts[0]
                        elif nx < 0 or nx >= self.cell_counts[0]:
                            continue
                        ncell = (nz * ny_cells + ny) * nx_cells + nx
                        results.append((ncell, self.cell_particle_count[ncell]))
        return results
    
    def kill_particle_index(self, pid: int) -> bool:
        """Kill particle by index."""
        return self._kill_impl(pid)
    
    def kill_random(self, count: int = 1) -> int:
        """Kill random particles uniformly."""
        killed = 0
        for _ in range(count):
            if self.population <= 0:
                break
            idx = int(np.random.randint(0, int(self.population)))
            if self._kill_impl(idx):
                killed += 1
        return killed
    
    def _kill_impl(self, pid: int) -> bool:
        """Internal kill implementation."""
        if pid < 0 or pid >= self.population:
            return False
        
        species = self.species_id[pid]
        cell_id = self.particle_cell[pid]
        slot = self.particle_slot[pid]
        
        pos = np.empty(self.ndim, dtype=np.float64)
        for d in range(self.ndim):
            pos[d] = self.positions[pid, d]
        
        pdr = self.death_rate[pid]
        self.total_birth_rate -= self.b[species]
        self.total_death_rate -= pdr
        self.cell_birth_rate[cell_id] -= self.b[species]
        self.cell_death_rate[cell_id] -= pdr
        self.cell_species_count[cell_id, species] -= 1
        
        # Remove interactions
        self._compute_interactions_kill(pid, species, cell_id, pos)
        
        # Remove from cell
        last_slot = self.cell_particle_count[cell_id] - 1
        last_p = self.cell_particles[cell_id, last_slot]
        self.cell_particles[cell_id, last_slot] = -1
        self.cell_particle_count[cell_id] = last_slot
        if slot != last_slot and last_p != -1:
            self.cell_particles[cell_id, slot] = last_p
            self.particle_slot[last_p] = slot
        
        # Compact arrays
        last_idx = int(self.population) - 1
        self.population -= 1
        if pid != last_idx:
            lc = self.particle_cell[last_idx]
            ls = self.particle_slot[last_idx]
            for d in range(self.ndim):
                self.positions[pid, d] = self.positions[last_idx, d]
            self.death_rate[pid] = self.death_rate[last_idx]
            self.species_id[pid] = self.species_id[last_idx]
            self.particle_cell[pid] = lc
            self.particle_slot[pid] = ls
            self.cell_particles[lc, ls] = pid
        
        # Clear last
        for d in range(self.ndim):
            self.positions[last_idx, d] = 0.0
        self.death_rate[last_idx] = 0.0
        self.species_id[last_idx] = -1
        self.particle_cell[last_idx] = -1
        self.particle_slot[last_idx] = -1
        
        if self.total_birth_rate < 0.0 and self.total_birth_rate > -1e-9:
            self.total_birth_rate = 0.0
        if self.total_death_rate < 0.0 and self.total_death_rate > -1e-9:
            self.total_death_rate = 0.0
        
        return True
    
    def _compute_interactions_kill(self, pid: int, species: int, cell_id: int, pos: NDArray) -> None:
        """Remove death interactions when killing."""
        for osp in range(self.species_count):
            cutoff_val = self.cutoff[species, osp]
            dd_val = self.dd[species, osp]
            if dd_val == 0.0 or cutoff_val <= 0.0:
                continue
            cull = self.cull_range[species, osp]
            for ncell, ncount in self._iter_neighbors(cell_id, cull):
                for s in range(ncount):
                    oid = self.cell_particles[ncell, s]
                    if oid == -1 or oid == pid:
                        continue
                    if self.species_id[oid] != osp:
                        continue
                    opos = self.positions[oid]
                    dist = _distance_nd(pos, opos, self.area_size, self.ndim, self.periodic)
                    if dist <= cutoff_val:
                        sigma = self.death_std[species, osp]
                        delta = dd_val * _gaussian_kernel(dist, sigma, self.ndim)
                        self.death_rate[oid] -= delta
                        self.cell_death_rate[ncell] -= delta
                        self.total_death_rate -= delta
    
    def _sample_parent(self, cell_id: int, species: int) -> int:
        """Sample random parent of species in cell."""
        cnt = self.cell_species_count[cell_id, species]
        if cnt <= 0:
            return -1
        target = int(np.random.random() * cnt)
        cc = self.cell_particle_count[cell_id]
        for s in range(cc):
            p = self.cell_particles[cell_id, s]
            if p == -1:
                continue
            if self.species_id[p] == species:
                if target == 0:
                    return p
                target -= 1
        return -1
    
    def _sample_victim(self, cell_id: int, species: int) -> int:
        """Sample victim weighted by death rate."""
        total = 0.0
        cc = self.cell_particle_count[cell_id]
        for s in range(cc):
            p = self.cell_particles[cell_id, s]
            if p != -1 and self.species_id[p] == species:
                total += self.death_rate[p]
        if total <= 0.0:
            return -1
        r = np.random.random() * total
        for s in range(cc):
            p = self.cell_particles[cell_id, s]
            if p != -1 and self.species_id[p] == species:
                r -= self.death_rate[p]
                if r <= 0.0:
                    return p
        return -1
    
    def attempt_birth_event(self) -> bool:
        """Attempt birth event."""
        cell_id = _sample_weighted(self.cell_birth_rate)
        if cell_id < 0:
            return False
        
        # Sample species from cell
        species_rates = np.empty(self.species_count, dtype=np.float64)
        for sp in range(self.species_count):
            species_rates[sp] = self.b[sp] * self.cell_species_count[cell_id, sp]
        species = _sample_weighted(species_rates)
        if species < 0:
            return False
        
        parent = self._sample_parent(cell_id, species)
        if parent < 0:
            return False
        
        # Normal dispersal
        child_pos = np.empty(self.ndim, dtype=np.float64)
        std = self.birth_std[species]
        for d in range(self.ndim):
            child_pos[d] = self.positions[parent, d] + np.random.normal(0.0, std)
        
        return self._spawn_impl(species, child_pos)
    
    def attempt_death_event(self) -> bool:
        """Attempt death event."""
        cell_id = _sample_weighted(self.cell_death_rate)
        if cell_id < 0:
            return False
        
        # Sample species from cell by death rate
        species_rates = np.empty(self.species_count, dtype=np.float64)
        cc = self.cell_particle_count[cell_id]
        for sp in range(self.species_count):
            r = 0.0
            for s in range(cc):
                p = self.cell_particles[cell_id, s]
                if p != -1 and self.species_id[p] == sp:
                    r += self.death_rate[p]
            species_rates[sp] = r
        species = _sample_weighted(species_rates)
        if species < 0:
            return False
        
        victim = self._sample_victim(cell_id, species)
        if victim < 0:
            return False
        
        return self._kill_impl(victim)
    
    def run_events(self, max_events: int) -> int:
        """Run specified number of events."""
        if max_events <= 0:
            return 0
        performed = 0
        for _ in range(max_events):
            total = self.total_birth_rate + self.total_death_rate
            if total <= 1e-12:
                break
            dt = -math.log(np.random.random()) / total
            self.time += dt
            if np.random.random() * total < self.total_birth_rate:
                if self.attempt_birth_event():
                    performed += 1
                    self.event_count += 1
            else:
                if self.attempt_death_event():
                    performed += 1
                    self.event_count += 1
        return performed
    
    def run_until_time(self, duration: float) -> int:
        """Run until time duration."""
        if duration <= 0.0:
            return 0
        target = self.time + duration
        performed = 0
        while self.time < target:
            total = self.total_birth_rate + self.total_death_rate
            if total <= 1e-12:
                break
            dt = -math.log(np.random.random()) / total
            if self.time + dt > target:
                self.time = target
                break
            self.time += dt
            if np.random.random() * total < self.total_birth_rate:
                if self.attempt_birth_event():
                    performed += 1
                    self.event_count += 1
            else:
                if self.attempt_death_event():
                    performed += 1
                    self.event_count += 1
        return performed
    
    def current_population(self) -> int:
        return int(self.population)
    
    def current_time(self) -> float:
        return self.time
    
    def get_species_counts(self) -> NDArray[np.int32]:
        counts = np.zeros(self.species_count, dtype=np.int32)
        for i in range(int(self.population)):
            sp = self.species_id[i]
            if sp >= 0:
                counts[sp] += 1
        return counts
    
    def reseed(self, seed: int) -> None:
        np.random.seed(int(seed))


def _calc_cell_counts(ndim: int, area: NDArray, max_cutoff: float, min_cells: int = 10) -> NDArray:
    """Calculate cell counts based on max cutoff."""
    if max_cutoff <= 0.0:
        max_cutoff = 1.0
    counts = np.empty(ndim, dtype=np.int32)
    for d in range(ndim):
        c = int(area[d] / max_cutoff)
        counts[d] = max(c, min_cells)
    return counts


def make_normal_ssa_1d(
    M: int, area_len: float,
    birth_rates: Sequence[float], death_rates: Sequence[float],
    dd_matrix: Sequence[Sequence[float]],
    birth_std: Sequence[float], death_std: Sequence[Sequence[float]],
    *, death_cull_sigmas: float = 5.0,
    cell_count: int | None = None, is_periodic: bool = False,
    seed: int | None = None,
) -> SSANormalState:
    """Create 1D Normal SSA state."""
    n = int(M)
    b = np.ascontiguousarray(birth_rates, dtype=np.float64)
    d = np.ascontiguousarray(death_rates, dtype=np.float64)
    dd = np.ascontiguousarray(dd_matrix, dtype=np.float64)
    bs = np.ascontiguousarray(birth_std, dtype=np.float64)
    ds = np.ascontiguousarray(death_std, dtype=np.float64)
    area = np.array([float(area_len)], dtype=np.float64)
    
    max_cut = np.max(ds) * death_cull_sigmas
    if cell_count is None:
        cells = _calc_cell_counts(1, area, max_cut)
    else:
        cells = np.array([int(cell_count)], dtype=np.int32)
    
    total_cells = int(np.prod(cells))
    capacity = max(n * total_cells * 10, 10000)
    cell_cap = max(32, capacity // total_cells * 2)
    
    return SSANormalState(
        np.int32(1), np.int32(n), area, cells, is_periodic,
        np.int64(capacity), np.int32(cell_cap),
        np.int64(seed if seed is not None else -1),
        b, d, dd, bs, ds, death_cull_sigmas
    )


def make_normal_ssa_2d(
    M: int, area_x: float, area_y: float,
    birth_rates: Sequence[float], death_rates: Sequence[float],
    dd_matrix: Sequence[Sequence[float]],
    birth_std: Sequence[float], death_std: Sequence[Sequence[float]],
    *, death_cull_sigmas: float = 5.0,
    cell_counts: tuple[int, int] | None = None, is_periodic: bool = False,
    seed: int | None = None,
) -> SSANormalState:
    """Create 2D Normal SSA state."""
    n = int(M)
    b = np.ascontiguousarray(birth_rates, dtype=np.float64)
    d = np.ascontiguousarray(death_rates, dtype=np.float64)
    dd = np.ascontiguousarray(dd_matrix, dtype=np.float64)
    bs = np.ascontiguousarray(birth_std, dtype=np.float64)
    ds = np.ascontiguousarray(death_std, dtype=np.float64)
    area = np.array([float(area_x), float(area_y)], dtype=np.float64)
    
    max_cut = np.max(ds) * death_cull_sigmas
    if cell_counts is None:
        cells = _calc_cell_counts(2, area, max_cut)
    else:
        cells = np.array([int(cell_counts[0]), int(cell_counts[1])], dtype=np.int32)
    
    total_cells = int(np.prod(cells))
    capacity = max(n * total_cells * 10, 10000)
    cell_cap = max(32, capacity // total_cells * 2)
    
    return SSANormalState(
        np.int32(2), np.int32(n), area, cells, is_periodic,
        np.int64(capacity), np.int32(cell_cap),
        np.int64(seed if seed is not None else -1),
        b, d, dd, bs, ds, death_cull_sigmas
    )


def make_normal_ssa_3d(
    M: int, area_x: float, area_y: float, area_z: float,
    birth_rates: Sequence[float], death_rates: Sequence[float],
    dd_matrix: Sequence[Sequence[float]],
    birth_std: Sequence[float], death_std: Sequence[Sequence[float]],
    *, death_cull_sigmas: float = 5.0,
    cell_counts: tuple[int, int, int] | None = None, is_periodic: bool = False,
    seed: int | None = None,
) -> SSANormalState:
    """Create 3D Normal SSA state."""
    n = int(M)
    b = np.ascontiguousarray(birth_rates, dtype=np.float64)
    d = np.ascontiguousarray(death_rates, dtype=np.float64)
    dd = np.ascontiguousarray(dd_matrix, dtype=np.float64)
    bs = np.ascontiguousarray(birth_std, dtype=np.float64)
    ds = np.ascontiguousarray(death_std, dtype=np.float64)
    area = np.array([float(area_x), float(area_y), float(area_z)], dtype=np.float64)
    
    max_cut = np.max(ds) * death_cull_sigmas
    if cell_counts is None:
        cells = _calc_cell_counts(3, area, max_cut)
    else:
        cells = np.array([int(cell_counts[0]), int(cell_counts[1]), int(cell_counts[2])], dtype=np.int32)
    
    total_cells = int(np.prod(cells))
    capacity = max(n * total_cells * 10, 10000)
    cell_cap = max(32, capacity // total_cells * 2)
    
    return SSANormalState(
        np.int32(3), np.int32(n), area, cells, is_periodic,
        np.int64(capacity), np.int32(cell_cap),
        np.int64(seed if seed is not None else -1),
        b, d, dd, bs, ds, death_cull_sigmas
    )


def get_all_particle_coords(state: SSANormalState) -> list[NDArray[np.float64]]:
    """Get particle coordinates for each species."""
    coords = []
    for sp in range(int(state.species_count)):
        collected = []
        for i in range(int(state.population)):
            if state.species_id[i] == sp:
                pos = np.empty(state.ndim, dtype=np.float64)
                for d in range(state.ndim):
                    pos[d] = state.positions[i, d]
                collected.append(pos)
        if collected:
            coords.append(np.array(collected, dtype=np.float64))
        else:
            coords.append(np.empty((0, state.ndim), dtype=np.float64))
    return coords


__all__ = [
    "SSANormalState",
    "make_normal_ssa_1d",
    "make_normal_ssa_2d",
    "make_normal_ssa_3d",
    "get_all_particle_coords",
]
