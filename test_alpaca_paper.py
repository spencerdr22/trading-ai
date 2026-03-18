"""
Test Alpaca paper trading connection.
"""

import asyncio
from app.execution.alpaca_paper import AlpacaPaperTrading


async def test_alpaca():
    """Test all Alpaca paper trading functions."""
    print("="*60)
    print("TESTING ALPACA PAPER TRADING")
    print("="*60)
    print()
    
    # Initialize client
    client = AlpacaPaperTrading()
    
    # Test 1: Get account info
    print("[1/3] Getting account information...")
    account = await client.get_account()
    if account:
        print(f"  ✅ Portfolio Value: ${float(account['portfolio_value']):,.2f}")
        print(f"  ✅ Buying Power: ${float(account['buying_power']):,.2f}")
        print(f"  ✅ Cash: ${float(account['cash']):,.2f}")
    print()
    
    # Test 2: Get positions
    print("[2/3] Getting current positions...")
    positions = await client.get_positions()
    if positions:
        for pos in positions:
            print(f"  📊 {pos['symbol']}: {pos['qty']} shares @ ${float(pos['avg_entry_price']):.2f}")
    else:
        print("  ℹ️  No open positions")
    print()
    
    # Test 3: Paper trade test (1 share of SPY)
    print("[3/3] Testing order placement (1 share SPY - paper money)...")
    confirm = input("  Place test order? (y/n): ")
    if confirm.lower() == 'y':
        order = await client.place_order(
            symbol="SPY",
            qty=1,
            side="buy",
            order_type="market"
        )
        if order:
            print(f"  ✅ Order ID: {order['id']}")
            print(f"  ✅ Status: {order['status']}")
    else:
        print("  ⏭️  Skipped")
    
    print()
    print("="*60)
    print("TEST COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_alpaca())
