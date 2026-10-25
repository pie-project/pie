use poem::Stream;
use poem::generative::{AxisRole, Generative, PositionConvention, ReadingFact, ReadoutKind};
use runtime::model::{validate_generative, velocity_facts};

fn text() -> ReadingFact {
    ReadingFact {
        name: "text".to_string(),
        index: 0,
        has_kv: true,
        takes_tokens: true,
        streams: Vec::new(),
        positions: None,
        readout: ReadoutKind::Hidden,
        readout_width: 512,
    }
}

fn denoise() -> ReadingFact {
    ReadingFact {
        name: "denoise".to_string(),
        index: 1,
        has_kv: false,
        takes_tokens: false,
        streams: vec![Stream::Image],
        positions: Some(PositionConvention {
            axes: vec![AxisRole::Time, AxisRole::Height, AxisRole::Width],
            text_axis: 0,
            text_origin: 0,
            image_follows_text: false,
            reference_stride: None,
        }),
        readout: ReadoutKind::Velocity,
        readout_width: 64,
    }
}

fn family(readings: Vec<ReadingFact>) -> Generative {
    Generative {
        readings,
        latent: None,
        schedule: None,
    }
}

#[test]
fn a_reading_family_is_dense_and_named_once_every_case() {
    a_well_formed_family_is_accepted_and_states_its_velocity();
    a_family_out_of_index_order_is_refused();
    duplicate_names_are_refused();
    readings_that_disagree_on_the_velocity_width_are_refused();
}

fn a_well_formed_family_is_accepted_and_states_its_velocity() {
    let both = family(vec![text(), denoise()]);
    assert_eq!(validate_generative(&both), Ok(()));
    assert_eq!(velocity_facts(&both.readings), (true, 64));
    assert_eq!(velocity_facts(&[text()]), (false, 0));
    assert_eq!(velocity_facts(&[]), (false, 0));
}

fn a_family_out_of_index_order_is_refused() {
    let mut swapped = denoise();
    swapped.index = 0;
    let why = validate_generative(&family(vec![text(), swapped])).unwrap_err();
    assert!(why.contains("`denoise`") && why.contains("index"), "{why}");

    let mut text_at_one = text();
    text_at_one.index = 1;
    assert!(validate_generative(&family(vec![text_at_one])).is_err());
}

fn duplicate_names_are_refused() {
    let mut second = text();
    second.index = 1;
    let why = validate_generative(&family(vec![text(), second])).unwrap_err();
    assert!(why.contains("declared twice"), "{why}");
}

fn readings_that_disagree_on_the_velocity_width_are_refused() {
    let mut low = denoise();
    low.name = "denoise.low".to_string();
    low.index = 2;
    low.readout_width = 128;
    let why = validate_generative(&family(vec![text(), denoise(), low])).unwrap_err();
    assert!(why.contains("velocity"), "{why}");
}
