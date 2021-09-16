"""
searching methods
"""

from typing import Union, Tuple, List


def linear_search(arr: List[Union[int, float, str]], ele: Union[int, float, str]) -> Union[Tuple[bool, int], Tuple[bool, None]]:
    """Linear search algorithm

    Args:
        arr (List[Union[int,float,str]]): array list to search
        ele (Union[int, float, str]): element to search

    Returns:
        Union[Tuple[bool, int], Tuple[bool, None]]: result of linear search
    """
    for i in range(len(arr)):
        if arr[i] == ele:
            return True, i

    return False, None


def binary_search(arr: List[Union[int, float, str]], ele: Union[int, float, str]) -> Union[Tuple[bool, int], Tuple[bool, None]]:
    """Binary search algorithm

    Args:
        arr (List[Union[int,float,str]]): array list to search
        ele (Union[int, float, str]): element to search

    Returns:
        Union[Tuple[bool, int], Tuple[bool, None]]: result of bianry search
    """
    low: int = 0
    high: int = len(arr) - 1

    while low <= high:
        mid = int(low + (high - low) / 2)
        if arr[mid] == ele:
            return True, mid
            
        if arr[mid] < ele:
            low = mid + 1
        else:
            high = mid - 1
    return False, None