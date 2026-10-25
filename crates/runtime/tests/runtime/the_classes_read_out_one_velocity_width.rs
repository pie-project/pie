use poem_ir::{LaneClass, LaneClasses, Readout, ReadoutKind};
use runtime::model::{pixels_width, velocity_width};

fn class(word: u64, readout: Option<(ReadoutKind, u32)>) -> LaneClass {
    LaneClass {
        words: vec![word],
        reads_kv: false,
        embeds: false,
        wide: false,
        readout: readout.map(|(kind, width)| Readout { kind, width }),
    }
}

fn classes(classes: Vec<LaneClass>) -> LaneClasses {
    LaneClasses {
        mask: 0b11,
        classes,
    }
}

#[test]
fn the_classes_read_out_one_velocity_width_every_case() {
    a_model_states_the_velocity_its_classes_read_out();
    classes_that_disagree_on_the_velocity_width_are_refused();
    pixels_of_two_widths_state_none();
}

fn a_model_states_the_velocity_its_classes_read_out() {
    let both = classes(vec![
        class(0, Some((ReadoutKind::Hidden, 512))),
        class(1, Some((ReadoutKind::Velocity, 64))),
        class(2, Some((ReadoutKind::Velocity, 64))),
    ]);
    assert_eq!(velocity_width(&both), Ok(Some(64)));
    assert_eq!(
        both.of(0b101).map(|c| c.words[0]),
        Some(1),
        "under the mask"
    );
    let text = classes(vec![class(0, Some((ReadoutKind::Logits, 1000)))]);
    assert_eq!(velocity_width(&text), Ok(None));
    assert_eq!(pixels_width(&text), None);
}

fn classes_that_disagree_on_the_velocity_width_are_refused() {
    let two = classes(vec![
        class(0, Some((ReadoutKind::Velocity, 64))),
        class(1, Some((ReadoutKind::Velocity, 128))),
    ]);
    let why = velocity_width(&two).unwrap_err();
    assert!(why.contains("velocity"), "{why}");
}

fn pixels_of_two_widths_state_none() {
    let two = classes(vec![
        class(0, Some((ReadoutKind::Pixels, 3))),
        class(1, Some((ReadoutKind::Pixels, 16))),
        class(2, None),
    ]);
    assert_eq!(pixels_width(&two), Some(0));
}
