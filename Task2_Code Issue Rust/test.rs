#[cfg(test)]
mod tests {
    use super::*; // Import items from the parent module

    #[test]
    fn test_round_trip() {
        // This test is highly comprehensive. It iterates through every single valid ID
        // from 0 up to N_STATES - 1. For each ID, it converts it to a boolean permutation
        // and then converts that permutation back to an ID.
        // It asserts that the final ID matches the original ID, ensuring that
        // permutation_to_id and id_to_permutation are perfect inverses of each other
        // across the entire valid range.
        for i in 0..N_STATES {
            let perm = id_to_permutation(i);
            let id = permutation_to_id(&perm);
            assert_eq!(
                id,
                i,
                "Round trip failed for ID: {}. Permutation: {:?}. Resulting ID: {}",
                i, perm, id
            );
        }
    }

    #[test]
    fn test_n_choose_k() {
        // Test various common cases for the binomial coefficient function.
        assert_eq!(n_choose_k(5, 2), 10, "C(5,2) should be 10");
        assert_eq!(n_choose_k(10, 3), 120, "C(10,3) should be 120");
        assert_eq!(
            n_choose_k(15, 7),
            6435,
            "C(15,7) should be 6435 (N_STATES)"
        );

        // Test with a smaller example (like the one used in manual verification).
        assert_eq!(n_choose_k(4, 2), 6, "C(4,2) should be 6");

        // Test edge cases where k is 0 or k is equal to n.
        assert_eq!(n_choose_k(5, 0), 1, "C(5,0) should be 1");
        assert_eq!(n_choose_k(5, 5), 1, "C(5,5) should be 1");

        // Test an invalid case where k is greater than n.
        assert_eq!(n_choose_k(3, 4), 0, "C(3,4) should be 0 (k > n)");
    }

    #[test]
    fn test_specific_permutations() {
        // These tests use a smaller N (e.g., 4 positions, 2 trues, N_STATES = 6)
        // to verify specific known mappings for easier debugging and understanding.

        // Test the lexicographically largest combination for 4 positions, 2 trues: [F,F,T,T] (indices 2,3) -> ID 0
        let perm_0 = [false, false, true, true];
        assert_eq!(permutation_to_id(&perm_0), 0, "Permutation [F,F,T,T] should map to ID 0");
        assert_eq!(id_to_permutation(0), perm_0, "ID 0 should map to [F,F,T,T]");

        // Test the lexicographically smallest combination for 4 positions, 2 trues: [T,T,F,F] (indices 0,1) -> ID 5
        let perm_5 = [true, true, false, false];
        assert_eq!(permutation_to_id(&perm_5), 5, "Permutation [T,T,F,F] should map to ID 5");
        assert_eq!(id_to_permutation(5), perm_5, "ID 5 should map to [T,T,F,F]");

        // Test an intermediate permutation: [F,T,F,T] (indices 1,3) -> ID 1
        let perm_1 = [false, true, false, true];
        assert_eq!(permutation_to_id(&perm_1), 1, "Permutation [F,T,F,T] should map to ID 1");
        assert_eq!(id_to_permutation(1), perm_1, "ID 1 should map to [F,T,F,T]");

        // Test another intermediate permutation: [T,F,T,F] (indices 0,2) -> ID 4
        let perm_4 = [true, false, true, false];
        assert_eq!(permutation_to_id(&perm_4), 4, "Permutation [T,F,T,F] should map to ID 4");
        assert_eq!(id_to_permutation(4), perm_4, "ID 4 should map to [T,F,T,F]");
    }
}