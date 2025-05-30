from typing import List, Dict, Any, Optional
from agentsociety.logger import get_logger


class FundManager:
    """
    Manages the funds for an environmental agent.
    
    - **Description**:
        - Handles all fund-related operations including tracking, updates, and history.
        - Provides protected access to funds with monitoring capabilities.
        
    - **Args**:
        - `initial_amount` (int): The initial amount of funds.
    """
    
    def __init__(self, initial_amount: int = 100000):
        """Initialize the fund manager with the initial amount."""
        self._funds = initial_amount
        self._funds_history = []
        
    @property
    def funds(self) -> int:
        """
        Get the current funds.
        
        - **Description**:
            - Returns the current amount of funds.
            - Funds cannot be directly modified.
            
        - **Returns**:
            - `int`: The current amount of funds.
        """
        return self._funds
        
    async def update_funds(self, amount: int, reason: Optional[str] = None) -> bool:
        """
        Update the funds with monitoring.
        
        - **Description**:
            - Updates the funds and records the transaction in history.
            - This is a protected method that should only be called by authorized methods.
            
        - **Args**:
            - `amount` (int): The amount to add/subtract from funds.
            - `reason` (str): The reason for the funds update.
        """
        if amount < 0:
            get_logger().warning(f"FundManager: Negative amount {amount} not allowed")
            return False
        if self._funds < amount:
            get_logger().warning(f"FundManager: Insufficient funds {self._funds} < {amount}")
            return False
        self._funds -= amount
        self._funds_history.append({
            'amount': amount,
            'reason': reason,
            'new_balance': self._funds
        })
        get_logger().info(f"FundManager: Spend {amount} funds for [{reason}]")
        return True
        
    def get_funds_history(self) -> List[Dict[str, Any]]:
        """
        Get the funds transaction history.
        
        - **Description**:
            - Returns the history of all funds transactions.
            
        - **Returns**:
            - `List[Dict[str, Any]]`: The funds transaction history.
        """
        return self._funds_history.copy()