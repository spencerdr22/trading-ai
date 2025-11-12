def pnl_per_tick(entry, exit, side):
    if side == "LONG":
        return exit - entry
    else:
        return entry - exit
