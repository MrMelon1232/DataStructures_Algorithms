"""
We're going to explore recursive backtracking
the demo will be done in python, familiarizing the notion of these data structures through python
"""

# formula for any recursive backtracking problem
def backtrack(current_state, choices_left):
    if base_case_condition:      # Base case
        save_solution(current_state)
        return

    for choice in choices_left:  # Loop through choices
        make_a_choice()          # Modify the state
        backtrack(new_state)     # Recurse
        undo_choice()            # Backtrack
