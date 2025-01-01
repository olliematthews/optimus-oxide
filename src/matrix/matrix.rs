use std::{
    collections::HashSet,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign},
};

use anyhow::Result;

use crate::{exceptions::MatrixError, matrix::vector::FloatVector};

pub struct Matrix2<const N_ROWS: usize, const N_COLS: usize> {
    pub rows: [FloatVector<N_COLS>; N_ROWS],
}

impl<const N_COLS: usize, const N_ROWS: usize> Matrix2<N_ROWS, N_COLS> {
    pub fn new_zeros() -> Self {
        Self::from_elements([[0.; N_COLS]; N_ROWS])
    }

    pub fn from_rows(rows: [FloatVector<N_COLS>; N_ROWS]) -> Self {
        Matrix2 { rows }
    }

    pub fn from_elements(elements: [[f32; N_COLS]; N_ROWS]) -> Self {
        let mut rows: [FloatVector<N_COLS>; N_ROWS] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..N_ROWS {
            rows[i] = FloatVector::from_elements(elements[i]);
        }
        Matrix2 { rows }
    }

    pub fn dot(&self, other: &FloatVector<N_COLS>) -> FloatVector<N_ROWS> {
        let mut ret_elements: [f32; N_ROWS] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..N_ROWS {
            ret_elements[i] = self.rows[i].dot(other);
        }
        FloatVector::from_elements(ret_elements)
    }
}

impl<const N_ROWS: usize, const N_COLS: usize> Add<&Self> for Matrix2<N_ROWS, N_COLS> {
    type Output = Matrix2<N_ROWS, N_COLS>;

    fn add(self, other: &Self::Output) -> Self::Output {
        let mut ret_elements: [FloatVector<N_COLS>; N_ROWS] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..N_ROWS {
            ret_elements[i] = &self.rows[i] + &other.rows[i]
        }
        Matrix2::from_rows(ret_elements)
    }
}

impl<const N_ROWS: usize, const N_COLS: usize> AddAssign<&Self> for Matrix2<N_ROWS, N_COLS> {
    fn add_assign(&mut self, other: &Self) {
        for i in 0..N_ROWS {
            self.rows[i] += &other.rows[i];
        }
    }
}

impl<const N_ROWS: usize, const N_COLS: usize, T> Mul<T> for &Matrix2<N_ROWS, N_COLS>
where
    T: Into<f32> + Copy,
{
    type Output = Matrix2<N_ROWS, N_COLS>;

    fn mul(self, other: T) -> Matrix2<N_ROWS, N_COLS> {
        let mut ret_rows: [FloatVector<N_COLS>; N_ROWS] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..N_ROWS {
            ret_rows[i] = &self.rows[i] * other
        }
        Matrix2::from_rows(ret_rows)
    }
}

impl<const N_ROWS: usize, const N_COLS: usize, T> MulAssign<T> for Matrix2<N_ROWS, N_COLS>
where
    T: Into<f32> + Copy,
{
    fn mul_assign(&mut self, other: T) {
        for row in &mut self.rows {
            *row *= other
        }
    }
}

impl<const N_ROWS: usize, const N_COLS: usize, T> Div<T> for &Matrix2<N_ROWS, N_COLS>
where
    T: Into<f32> + Copy,
{
    type Output = Matrix2<N_ROWS, N_COLS>;

    fn div(self, other: T) -> Matrix2<N_ROWS, N_COLS> {
        let mut ret_rows: [FloatVector<N_COLS>; N_ROWS] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..N_ROWS {
            ret_rows[i] = &self.rows[i] / other
        }
        Matrix2::from_rows(ret_rows)
    }
}

impl<const N_ROWS: usize, const N_COLS: usize, T> DivAssign<T> for Matrix2<N_ROWS, N_COLS>
where
    T: Into<f32> + Copy,
{
    fn div_assign(&mut self, other: T) {
        for row in &mut self.rows {
            *row /= other
        }
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
