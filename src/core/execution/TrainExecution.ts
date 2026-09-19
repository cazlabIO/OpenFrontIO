import {
  Execution,
  Game,
  Player,
  TrainType,
  Unit,
  UnitType,
} from "../game/Game";
import { TileRef } from "../game/GameMap";
import { MotionPlanRecord } from "../game/MotionPlans";
import { RailNetwork } from "../game/RailNetwork";
import { getOrientedRailroad, OrientedRailroad } from "../game/Railroad";
import { TrainStation } from "../game/TrainStation";

export class TrainExecution implements Execution {
  private active = true;
  private mg: Game | null = null;
  private train: Unit | null = null; // primary unit
  private cars: Unit[] = []; // stored back to front
  private hasCargo: boolean = false;
  private currentTile: number = 0;
  private spacing = 2;
  private usedTiles: TileRef[] = []; // used for cars behind
  private stations: TrainStation[] = [];
  private currentRailroad: OrientedRailroad | null = null;
  private speed: number = 2;
  private _tradeStopsVisited: number = 0;
  private pathTiles: TileRef[] = [];
  private pathIndex: number = 0;

  constructor(
    private railNetwork: RailNetwork,
    private player: Player,
    private source: TrainStation,
    private destination: TrainStation,
    private numCars: number,
  ) {}

  public owner(): Player {
    return this.player;
  }

  public tradeStopsVisited(): number {
    return this._tradeStopsVisited;
  }

  init(mg: Game, ticks: number): void {
    this.mg = mg;
    const stations = this.railNetwork.findStationsPath(
      this.source,
      this.destination,
    );
    if (!stations || stations.length <= 1) {
      this.active = false;
      return;
    }

    this.stations = stations;
    const railroad = getOrientedRailroad(this.stations[0], this.stations[1]);
    if (railroad) {
      this.currentRailroad = railroad;
    } else {
      this.active = false;
      return;
    }

    const spawn = this.player.canBuild(UnitType.Train, this.stations[0].tile());
    if (spawn === false) {
      console.warn(`cannot build train`);
      this.active = false;
      return;
    }
    this.train = this.createTrainUnits(spawn);

    const carUnitIds = this.cars.map((c) => c.id());
    const pathTiles: TileRef[] = [];
    for (let i = 0; i + 1 < this.stations.length; i++) {
      const segment = getOrientedRailroad(
        this.stations[i],
        this.stations[i + 1],
      );
      if (!segment) {
        this.active = false;
        return;
      }
      pathTiles.push(...segment.getTiles());
    }
    const startTile = this.train.tile();
    if (pathTiles.length === 0 || pathTiles[0] !== startTile) {
      pathTiles.unshift(startTile);
      this.pathIndex = 1;
    }
    this.pathTiles = pathTiles;

    const plan: MotionPlanRecord = {
      kind: "train",
      engineUnitId: this.train.id(),
      carUnitIds,
      planId: 1,
      startTick: ticks + 1,
      speed: this.speed,
      spacing: this.spacing,
      path: pathTiles,
    };
    this.mg.recordMotionPlan(plan);
  }

  tick(ticks: number): void {
    if (this.train === null) {
      throw new Error("Not initialized");
    }

    if (!this.train.isActive()) {
      this.deleteTrain();
      return;
    }

    // A station added to the railroad currently under the train replaces that
    // railroad with two (or more) segments. Rebase the train onto the segment
    // containing its engine before moving so stations still ahead are visited,
    // while stations already passed are not paid retroactively.
    this.reconcileCurrentRailroadSplit();

    if (!this.activeSourceOrDestination()) {
      this.deleteTrain();
      return;
    }

    const tile = this.getNextTile();
    if (tile) {
      this.updateCarsPositions(tile);
    } else {
      this.targetReached();
      this.deleteTrain();
    }
  }

  loadCargo() {
    if (this.hasCargo || this.train === null) {
      return;
    }
    this.hasCargo = true;
    // Starts at 1: don't load tail engine
    for (let i = 1; i < this.cars.length; i++) {
      this.cars[i].setLoaded(true);
    }
  }

  private targetReached() {
    if (this.train === null) {
      return;
    }
    this.train.setReachedTarget();
    this.cars.forEach((car: Unit) => {
      car.setReachedTarget();
    });
  }

  private createTrainUnits(tile: TileRef): Unit {
    const train = this.player.buildUnit(UnitType.Train, tile, {
      targetUnit: this.destination.unit,
      trainType: TrainType.Engine,
    });
    // Tail is also an engine, just for cosmetics
    this.cars.push(
      this.player.buildUnit(UnitType.Train, tile, {
        targetUnit: this.destination.unit,
        trainType: TrainType.TailEngine,
      }),
    );
    for (let i = 0; i < this.numCars; i++) {
      this.cars.push(
        this.player.buildUnit(UnitType.Train, tile, {
          trainType: TrainType.Carriage,
          loaded: this.hasCargo,
        }),
      );
    }
    return train;
  }

  private deleteTrain() {
    this.active = false;
    if (this.train?.isActive()) {
      this.train.delete(false);
    }
    for (const car of this.cars) {
      if (car.isActive()) {
        car.delete(false);
      }
    }
  }

  private activeSourceOrDestination(): boolean {
    return (
      this.stations.length > 1 &&
      this.stations[1].isActive() &&
      this.stations[0].isActive()
    );
  }

  /**
   * Save the tiles the train go through so the cars can reuse them
   * Don't simply save the tiles the engine uses, otherwise the spacing will be dictated by the train speed
   */
  private saveTraversedTiles(from: number, speed: number) {
    if (!this.currentRailroad) {
      return;
    }
    let tileToSave: number = from;
    for (
      let i = 0;
      i < speed && tileToSave < this.currentRailroad.getTiles().length;
      i++
    ) {
      this.saveTile(this.currentRailroad.getTiles()[tileToSave]);
      tileToSave = tileToSave + 1;
    }
  }

  private saveTile(tile: TileRef) {
    this.usedTiles.push(tile);
    if (this.usedTiles.length > this.cars.length * this.spacing + 3) {
      this.usedTiles.shift();
    }
  }

  private updateCarsPositions(newTile: TileRef) {
    if (this.cars.length > 0) {
      for (let i = this.cars.length - 1; i >= 0; --i) {
        const carTileIndex = (i + 1) * this.spacing + 2;
        if (this.usedTiles.length > carTileIndex) {
          this.cars[i].move(this.usedTiles[carTileIndex]);
        }
      }
    }
    if (this.train !== null) {
      this.train.move(newTile);
    }
  }

  private nextStation(): boolean {
    if (this.stations.length > 2) {
      this.pathIndex += this.currentRailroad?.getTiles().length ?? 0;
      this.stations.shift();
      const railRoad =
        getOrientedRailroad(this.stations[0], this.stations[1]) ??
        this.resolveSplitRailroad();
      if (railRoad) {
        this.currentRailroad = railRoad;
        return true;
      }
    }
    return false;
  }

  private resolveSplitRailroad(): OrientedRailroad | null {
    const [station0, station1] = this.stations;
    const path = this.railNetwork.findStationsPath(station0, station1);
    if (!path || path.length <= 2) return null;
    let cursor = this.pathIndex;
    for (let i = 0; i < path.length - 1; i++) {
      const segment = getOrientedRailroad(path[i], path[i + 1]);
      if (!segment) return null;
      for (const tile of segment.getTiles()) {
        if (this.pathTiles[cursor++] !== tile) {
          return null;
        }
      }
    }
    this.stations.splice(0, 2, ...path);
    return getOrientedRailroad(this.stations[0], this.stations[1]);
  }

  /**
   * Rebase a train when its current railroad was split by newly-added stations.
   *
   * Rail splitting preserves the original tile sequence. That lets us compare
   * the replacement segments with the recorded motion plan, then use
   * currentTile to select the replacement segment containing the engine. A
   * station exactly at currentTile remains the next station so the normal
   * boundary-crossing logic awards it once; stations behind currentTile are
   * discarded without a stop.
   */
  private reconcileCurrentRailroadSplit(): void {
    if (this.currentRailroad === null || this.stations.length < 2) return;

    const [station0, station1] = this.stations;
    if (getOrientedRailroad(station0, station1) !== null) return;

    const path = this.railNetwork.findStationsPath(station0, station1);
    if (!path || path.length <= 2) return;

    const segments: OrientedRailroad[] = [];
    let cursor = this.pathIndex;
    for (let i = 0; i < path.length - 1; i++) {
      const segment = getOrientedRailroad(path[i], path[i + 1]);
      if (!segment) return;
      segments.push(segment);
      for (const tile of segment.getTiles()) {
        if (this.pathTiles[cursor++] !== tile) return;
      }
    }

    // Only adopt a true split of the current railroad. A detour may share a
    // prefix with the old route, but must not change the train's motion plan.
    const oldLength = this.currentRailroad.getTiles().length;
    if (cursor !== this.pathIndex + oldLength) return;

    const oldTiles = this.currentRailroad.getTiles();
    let activeSegment = 0;
    let segmentStart = 0;
    let boundary = 0;
    for (let i = 0; i < segments.length - 1; i++) {
      boundary += segments[i].getTiles().length;
      const stopIndex = this.stopIndexAtBoundary(
        path[i + 1],
        oldTiles,
        boundary,
      );
      if (this.currentTile <= stopIndex) break;
      activeSegment = i + 1;
      segmentStart = boundary;
    }

    this.pathIndex += segmentStart;
    this.currentTile -= segmentStart;
    this.currentRailroad = segments[activeSegment];
    this.stations.splice(0, 2, ...path.slice(activeSegment));
  }

  /**
   * A split assigns the closest rail tile to one of the two new segments. The
   * assignment depends on railroad orientation, so the station can lie at
   * either side of the segment-array boundary. Pick the adjacent tile closest
   * to the station to decide whether the engine has physically passed it.
   */
  private stopIndexAtBoundary(
    station: TrainStation,
    tiles: readonly TileRef[],
    boundary: number,
  ): number {
    if (this.mg === null) throw new Error("Not initialized");

    const before = boundary - 1;
    const after = boundary;
    const stationX = this.mg.x(station.tile());
    const stationY = this.mg.y(station.tile());
    const distanceSquared = (index: number) => {
      const dx = this.mg!.x(tiles[index]) - stationX;
      const dy = this.mg!.y(tiles[index]) - stationY;
      return dx * dx + dy * dy;
    };

    return distanceSquared(before) <= distanceSquared(after) ? before : after;
  }

  private canTradeWithDestination() {
    return (
      this.stations.length > 1 && this.stations[1].tradeAvailable(this.player)
    );
  }

  private getNextTile(): TileRef | null {
    if (this.currentRailroad === null || !this.canTradeWithDestination()) {
      return null;
    }
    this.saveTraversedTiles(this.currentTile, this.speed);
    this.currentTile = this.currentTile + this.speed;
    const leftOver = this.currentTile - this.currentRailroad.getTiles().length;
    if (leftOver >= 0) {
      // Station reached, pick the next station
      this.stationReached();
      if (!this.nextStation()) {
        return null; // Destination reached (or no valid connection)
      }
      this.currentTile = leftOver;
      this.saveTraversedTiles(0, leftOver);
    }
    return this.currentRailroad.getTiles()[this.currentTile];
  }

  private stationReached() {
    if (this.mg === null || this.player === null) {
      throw new Error("Not initialized");
    }
    this.stations[1].onTrainStop(this);
    const stationType = this.stations[1].unit.type();
    if (stationType === UnitType.City || stationType === UnitType.Port) {
      this._tradeStopsVisited++;
    }
    return;
  }

  isActive(): boolean {
    return this.active;
  }

  activeDuringSpawnPhase(): boolean {
    return false;
  }
}
