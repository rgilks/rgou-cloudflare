use rgou_ai_core::roll_tetrahedral_dice;

#[test]
fn test_tetrahedral_dice_distribution() {
    println!("🎲 Testing Tetrahedral Dice Distribution");
    println!("{}", "=".repeat(50));

    const NUM_ROLLS: usize = 10000;
    let mut counts = [0; 5];

    for _ in 0..NUM_ROLLS {
        let roll = roll_tetrahedral_dice();
        counts[roll as usize] += 1;
    }

    println!("Roll distribution over {} rolls:", NUM_ROLLS);
    println!("Roll | Count | Percentage | Expected");
    println!("-----|-------|------------|---------");

    let expected_counts = [625, 2500, 3750, 2500, 625]; // 1/16, 4/16, 6/16, 4/16, 1/16
    let expected_percentages = [6.25, 25.0, 37.5, 25.0, 6.25];

    for i in 0..5 {
        let percentage = (counts[i] as f64 / NUM_ROLLS as f64) * 100.0;
        println!(
            "{:4} | {:5} | {:9.2}% | {:6.2}%",
            i, counts[i], percentage, expected_percentages[i]
        );
    }

    // Verify the distribution is reasonable (within 2% of expected)
    for i in 0..5 {
        let actual_percentage = (counts[i] as f64 / NUM_ROLLS as f64) * 100.0;
        let expected_percentage = expected_percentages[i];
        let difference = (actual_percentage - expected_percentage).abs();

        assert!(
            difference < 2.0,
            "Roll {}: {}% vs expected {}% (difference: {:.2}%)",
            i,
            actual_percentage,
            expected_percentage,
            difference
        );
    }

    println!("✅ Dice distribution is correct!");
}
