# Cold-chain dispatcher

You are the dispatcher for a refrigerated delivery fleet for one 480-minute shift. Deliver as many orders as possible on time before they spoil.

Control the fleet with the `dispatch` command (run `dispatch help`):

- `dispatch state` shows the clock, every vehicle (position, load, capacity, remaining route) and every order you can currently see.
- `dispatch route <vehicle> <stop> <stop> ...` replaces a vehicle's remaining stops. Stops are order ids or `DEPOT`. A vehicle's load drops by each order's demand, and visiting `DEPOT` refills it to capacity (takes 10 minutes). Routes that would exceed capacity are rejected.
- `dispatch wait <minutes>` advances the clock by 1–120 minutes. Vehicles only move while the clock advances.
- `dispatch finish` ends your shift. Any remaining time still passes with the routes as they are.

## Rules

- The depot is at (0, 0). Vehicles travel 1 distance unit per minute in straight lines.
- Each order has a delivery window `tw = [open, close]`. Arriving before `open` means waiting. Service takes 5 minutes.
- Each order has a `spoil_at` time. A delivery starting after it is worthless, and so is an order nobody delivers by then.
- **New orders appear during the shift.** You only see an order once it has been released.

## Scoring

Every order in the shift counts, including ones you never saw:

- delivered within its window: 1
- delivered late but before spoiling: 0.5
- spoiled or not delivered: 0

Your score is the average of these, reduced by up to 30% for total distance driven. It is between 0 and 1.

The shift ends at minute 480, or earlier if you call `dispatch finish`.
