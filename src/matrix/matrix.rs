use std::collections::HashSet;

use anyhow::Result;

use crate::{exceptions::MatrixError, matrix::vector::FloatVector};

pub struct Matrix2<const N_COLS: usize, const N_ROWS: usize> {
    pub rows: [FloatVector<N_COLS>; N_ROWS],
}

impl<const N_COLS: usize, const N_ROWS: usize> Matrix2<N_COLS, N_ROWS> {
    pub fn from_rows(rows: [FloatVector<N_COLS>; N_ROWS]) -> Self {
        Matrix2 { rows }
    }

    pub fn from_elements(elements: [[f32; N_COLS]; N_ROWS]) -> Self {
        let mut rows: [FloatVector<N_COLS>; N_ROWS] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..N_COLS {
            rows[i] = FloatVector::from_elements(elements[i]);
        }
        Matrix2 { rows }
    }

    pub fn dot(&self, other: &FloatVector<N_COLS>) -> FloatVector<N_COLS> {
        let mut ret_elements: [f32; N_COLS] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..N_COLS {
            ret_elements[i] = self.rows[i].dot(other);
        }
        FloatVector::from_elements(ret_elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot() {
        let A_iden = Matrix2::from_elements([[1., 0.], [0., 1.]]);
        let b = FloatVector::from_elements([5., 4.]);

        let c = A_iden.dot(&b);

        assert_eq!(c, b);

        let A_rot = Matrix2::from_elements([[0., -1.], [1., 0.]]);
        let c = A_rot.dot(&b);

        assert_eq!(c, FloatVector::from_elements([-4., 5.]));
    }
}
