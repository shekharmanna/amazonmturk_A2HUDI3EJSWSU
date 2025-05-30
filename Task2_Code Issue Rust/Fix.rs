use std::vec::Vec; // Explicitly import Vec for clarity
use std::iter::Rev; // Explicitly import Rev for clarity

// N_STATES represents the total number of unique permutations of 8 falses and 7 trues
// in an array of size 15. This is calculated as "15 choose 7" (C(15, 7)).
// C(15, 7) = 15! / (7! * (15-7)!) = 15! / (7! * 8!) = 6435.
const N_STATES: u64 = 6435;

// Get the unique id of a permutation of 8 falses and 7 trues.
// The permutation is represented as a boolean array of size 15.
// This function implements a ranking algorithm for combinations.
pub fn permutation_to_id(perm: &[bool; 15]) -> u64 {
    // Collect the 0-indexed positions of all 'true' values in the permutation.
    // For example, if perm is [F, F, T, F, T, ...], and the first 'T' is at index 2,
    // then 2 would be in 'indices'.
    let indices: Vec<u8> = perm
        .iter()
        .enumerate() // Gives (index, &boolean_value) pairs
        .filter_map(|(i, &b)| {
            // Keep only the elements where 'b' (the boolean value) is true.
            // Map the 'usize' index 'i' to 'u8'.
            if b { Some(i as u8) } else { None }
        })
        .collect(); // Collects the filtered indices into a Vec<u8>

    // Initialize 'sum' to N_STATES - 1. This is part of the specific ranking scheme
    // where the lexicographically largest combination (e.g., [..., F, T, T, T])
    // maps to 0, and the smallest (e.g., [T, T, T, ..., F]) maps to N_STATES - 1.
    let mut sum = N_STATES - 1;

    // Iterate through the collected 'indices' along with their 0-indexed position within 'indices'.
    // 'i' is the 0-indexed count of the 'true' value (0 for the 1st true, 1 for the 2nd, etc.)
    // 'idx' is the actual 0-indexed position of the 'true' value in the original 15-element array.
    for (i, &idx) in indices.iter().enumerate() {
        // This line implements the core ranking logic.
        // It subtracts n_choose_k(idx, i + 1) from the running sum.
        // 'idx' is 'n' in "n choose k", and 'i + 1' is 'k' (since 'i' is 0-indexed, 'k' starts from 1).
        // The sum of these binomial coefficients uniquely identifies the combination.
        sum -= n_choose_k(idx as u64, i as u64 + 1);
    }
    sum
}

// Get the permutation of 8 falses and 7 trues from the unique id.
// This function implements an unranking algorithm for combinations.
pub fn id_to_permutation(id: u64) -> [bool; 15] {
    // This is the crucial step to reverse the ranking.
    // The 'id' (the target rank) is transformed back into the sum of binomial coefficients
    // that corresponds to the desired combination. This 'target_sum' is what the
    // unranking algorithm will iteratively reduce.
    let mut target_sum = N_STATES - 1 - id;
    let mut array = [false; 15]; // Initialize a boolean array with all 'false'

    // Iterate backwards from the 7th 'true' value down to the 1st.
    // 'i' here represents the 0-indexed count of the 'true' value we are trying to find (from 6 down to 0).
    // So, 'i + 1' is the 'k' in the n_choose_k(n, k) formula.
    for i in (0..7).rev() {
        // Find the largest possible index 'idx' (position in the 15-element array)
        // such that 'n_choose_k(idx, i + 1)' is less than or equal to the 'target_sum'.
        // We search from right to left (index 14 down to 0) to ensure we pick the
        // correct combination that matches the lexicographical ordering.
        let index = (0u8..15)
            .rev() // Iterate from 14 down to 0
            .find(|&idx| {
                // Calculate n_choose_k for the current 'idx' and 'k' (i+1)
                // and check if it's less than or equal to the remaining target_sum.
                n_choose_k(idx as u64, i as u64 + 1) <= target_sum
            })
            .expect("Index must be found for unranking. This usually indicates an issue with the input ID or the n_choose_k logic, or that the ID is out of the valid range.");

        // Subtract the found binomial coefficient from 'target_sum'.
        // This accounts for the contribution of the current 'true' value's position
        // and reduces the sum needed for finding the remaining 'true' values.
        target_sum -= n_choose_k(index as u64, i as u64 + 1);

        // Mark this position as true in the permutation array.
        array[index as usize] = true;
    }
    array
}

// Calculate "n choose k" (binomial coefficient).
// This function computes C(n, k) = n! / (k! * (n-k)!).
// It's optimized to avoid large factorials and handles edge cases.
fn n_choose_k(mut n: u64, mut k: u64) -> u64 {
    // If k is greater than n, it's impossible to choose k items from n, so return 0.
    if k > n {
        return 0;
    }
    // Optimization: C(n, k) = C(n, n-k).
    // We choose the smaller value for 'k' to reduce the number of iterations.
    if k > n - k {
        k = n - k;
    }
    // If k is 0, C(n, 0) is always 1 (there's one way to choose 0 items: choose nothing).
    if k == 0 {
        return 1;
    }

    let mut c = 1; // Initialize result to 1

    // Iteratively calculate C(n, k)
    // The loop runs 'k' times.
    // In each iteration, 'c' is multiplied by 'n' and then divided by 'i'.
    // 'n' is decremented in each step. This avoids calculating large factorials
    // and ensures intermediate products remain manageable.
    for i in 1..=k {
        c = c * n / i;
        n -= 1;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_trip() {
        // This test iterates through all possible unique IDs (0 to N_STATES - 1).
        for i in 0..N_STATES {
            // Convert the ID to its boolean permutation.
            let perm = id_to_permutation(i);
            // Convert the boolean permutation back to an ID.
            let id = permutation_to_id(&perm);
            // Assert that the round trip results in the original ID.
            assert_eq!(id, i, "Round trip failed for ID: {}. Permutation: {:?}. Resulting ID: {}", i, perm, id);
        }
    }

    #[test]
    fn test_n_choose_k() {
        assert_eq!(n_choose_k(5, 2), 10);
        assert_eq!(n_choose_k(10, 3), 120);
        assert_eq!(n_choose_k(15, 7), 6435); // N_STATES
        assert_eq!(n_choose_k(4, 2), 6);
        assert_eq!(n_choose_k(5, 0), 1);
        assert_eq!(n_choose_k(5, 5), 1);
        assert_eq!(n_choose_k(3, 4), 0); // k > n
    }

    #[test]
    fn test_specific_permutations() {
        // Example: 4 positions, 2 trues (N_STATES = 6, IDs 0-5)
        // [F,F,T,T] (indices 2,3) -> ID 0
        let perm_0 = [false, false, true, true];
        assert_eq!(permutation_to_id(&perm_0), 0);
        assert_eq!(id_to_permutation(0), perm_0);

        // [T,T,F,F] (indices 0,1) -> ID 5
        let perm_5 = [true, true, false, false];
        assert_eq!(permutation_to_id(&perm_5), 5);
        assert_eq!(id_to_permutation(5), perm_5);

        // [F,T,F,T] (indices 1,3) -> ID 1
        let perm_1 = [false, true, false, true];
        assert_eq!(permutation_to_id(&perm_1), 1);
        assert_eq!(id_to_permutation(1), perm_1);

        // [T,F,T,F] (indices 0,2) -> ID 4
        let perm_4 = [true, false, true, false];
        assert_eq!(permutation_to_id(&perm_4), 4);
        assert_eq!(id_to_permutation(4), perm_4);
    }
}
