#[macro_export]
macro_rules! do_at_key_with_default {
    ($map:expr, $key:expr, push $element:expr) => {
        if let Some(elements) = $map.get_mut($key) {
            elements.push($element);
        } else {
            $map.insert($key.clone(), vec![$element]);
        }
    };
    ($map:expr, $key:expr, insert $element:expr) => {
        if let Some(elements) = $map.get_mut($key) {
            elements.insert($element);
        } else {
            $map.insert($key.clone(), HashSet::from([$element]));
        }
    };
    ($map:expr, $key:expr, add $to_add:expr, $default:expr) => {
        if let Some(value) = $map.get_mut($key) {
            *value += $to_add;
        } else {
            $map.insert($key.clone(), $default);
        }
    };
    ($map:expr, $key:expr, sub $to_sub:expr, $default:expr) => {
        if let Some(value) = $map.get_mut($key) {
            *value -= $to_sub;
        } else {
            $map.insert($key.clone(), $default);
        }
    };
}
