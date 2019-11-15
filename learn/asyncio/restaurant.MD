# Reference
[Ref: Uche Ogbuji's excellent tutorial on Coroutines and asyncio](https://developer.ibm.com/tutorials/ba-on-demand-data-python-3/)

# Scenario
There is a tiny restaurant. With three tables and only one server and the following process:
1. Server will come with a menu and return for your order
2. Once the cook has prepared the order, server will bring it to you
3. After eating is finished, server brings the check
4. Server returns to the table when you are ready for payment

The entire dining experience takes 1 hour

# Pseudocode for asyncio

```python
async def serve_table(table_number):
    await get_menus()
    print('Welcome. Please set at table.', table_number, 'Here are your menus.')
    order = await get_order()
    print('Table', table_number, 'What will you be having today?)
    await prepare_order()
    print('Table', table_number, 'Here is your meal')
    await eat()
    print('Table', table_number, 'Here is your check')
    await get_payment()
    print('Thanks for visiting us! (table', table_number', ')')
```
