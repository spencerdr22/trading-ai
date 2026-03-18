from app.execution.alpaca_paper import get_alpaca_client

c = get_alpaca_client()
a = c.get_account()
p = c.get_positions()

print("----------------------------")
print(f"Portfolio Value : ${float(a['portfolio_value']):,.2f}")
print(f"Cash            : ${float(a['cash']):,.2f}")
print(f"Buying Power    : ${float(a['buying_power']):,.2f}")
print(f"Open Positions  : {len(p)}")
for pos in p:
    print(f"  {pos['symbol']} | qty={pos['qty']} | side={pos['side']} | unrealized PnL=${float(pos['unrealized_pl']):,.2f}")
print("----------------------------")
