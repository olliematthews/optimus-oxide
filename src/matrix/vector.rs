use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use anyhow::Result;

use crate::exceptions::MatrixError;

use super::matrix::Matrix2;

#[derive(Debug, Clone, PartialEq)]
pub struct FloatVector<const SIZE: usize> {
    elements: [f32; SIZE],
}

impl<const SIZE: usize> FloatVector<SIZE> {
    pub fn new_zeros() -> Self {
        Self::from_elements([0.; SIZE])
    }

    pub fn from_vector(elements: Vec<f32>) -> Result<FloatVector<SIZE>> {
        Ok(FloatVector {
            elements: elements.try_into().map_err(|_| {
                MatrixError::MatrixError(String::from("Failed to convert Vec to array"))
            })?,
        })
    }

    pub fn from_elements(elements: [f32; SIZE]) -> FloatVector<SIZE> {
        FloatVector { elements }
    }

    pub fn from_slice(elements: &[f32]) -> Result<FloatVector<SIZE>> {
        Ok(FloatVector {
            elements: elements.try_into().map_err(|_| {
                MatrixError::MatrixError(String::from("Failed to convert Vec to array"))
            })?,
        })
    }

    pub fn len(&self) -> usize {
        SIZE
    }

    pub fn dot(&self, other: &FloatVector<SIZE>) -> f32 {
        self.elements
            .iter()
            .zip(other.elements.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    pub fn outer<const OTHER_SIZE: usize>(
        &self,
        other: &FloatVector<OTHER_SIZE>,
    ) -> Matrix2<SIZE, OTHER_SIZE> {
        let mut ret_rows: [[f32; OTHER_SIZE]; SIZE] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        // let ret_elements: Matrix2<OTHER_SIZE, SIZE> = {}
        self.elements.iter().enumerate().for_each(|(i, i_element)| {
            other
                .iter()
                .enumerate()
                .for_each(|(j, j_element)| ret_rows[i][j] = i_element * j_element);
        });
        Matrix2::from_elements(ret_rows)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, f32> {
        self.elements.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, f32> {
        self.elements.iter_mut()
    }
}

impl<const SIZE: usize> IntoIterator for FloatVector<SIZE> {
    type Item = f32;
    type IntoIter = std::array::IntoIter<Self::Item, SIZE>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.into_iter()
    }
}

impl<const SIZE: usize> FromIterator<f32> for FloatVector<SIZE> {
    fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
        let mut ret_elements: [f32; SIZE] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };

        let mut idx: usize = 0;
        for element in iter {
            ret_elements[idx] = element;
            idx += 1;
        }
        assert_eq!(idx, SIZE, "The iterator contained {} items, which is incorrect to create a FloatVector of size {}.", idx, SIZE);
        Self::from_elements(ret_elements)
    }
}

impl<'a, 'b, const SIZE: usize> Add<&'b FloatVector<SIZE>> for &'a FloatVector<SIZE> {
    type Output = FloatVector<SIZE>;

    fn add(self, other: &'b FloatVector<SIZE>) -> FloatVector<SIZE> {
        let mut ret_elements: [f32; SIZE] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..SIZE {
            ret_elements[i] = self.elements[i] + other.elements[i]
        }
        FloatVector::from_elements(ret_elements)
    }
}

impl<const SIZE: usize> AddAssign<&Self> for FloatVector<SIZE> {
    fn add_assign(&mut self, other: &Self) {
        for i in 0..SIZE {
            self.elements[i] = self.elements[i] + other.elements[i];
        }
    }
}

impl<'a, 'b, const SIZE: usize> Sub<&'b FloatVector<SIZE>> for &'a FloatVector<SIZE> {
    type Output = FloatVector<SIZE>;

    fn sub(self, other: &'b FloatVector<SIZE>) -> FloatVector<SIZE> {
        let mut ret_elements: [f32; SIZE] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..SIZE {
            ret_elements[i] = self.elements[i] - other.elements[i]
        }
        FloatVector::from_elements(ret_elements)
    }
}

impl<const SIZE: usize> SubAssign<&Self> for FloatVector<SIZE> {
    fn sub_assign(&mut self, other: &Self) {
        for i in 0..SIZE {
            self.elements[i] = self.elements[i] - other.elements[i];
        }
    }
}

impl<const SIZE: usize, T> Mul<T> for &FloatVector<SIZE>
where
    T: Into<f32> + Copy,
{
    type Output = FloatVector<SIZE>;

    fn mul(self, other: T) -> FloatVector<SIZE> {
        let mut ret_elements: [f32; SIZE] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..SIZE {
            ret_elements[i] = self.elements[i] * other.into()
        }
        FloatVector::from_elements(ret_elements)
    }
}

impl<const SIZE: usize, T> MulAssign<T> for FloatVector<SIZE>
where
    T: Into<f32> + Copy,
{
    fn mul_assign(&mut self, other: T) {
        for i in 0..SIZE {
            self.elements[i] *= other.into()
        }
    }
}

impl<const SIZE: usize, T> Div<T> for &FloatVector<SIZE>
where
    T: Into<f32> + Copy,
{
    type Output = FloatVector<SIZE>;

    fn div(self, other: T) -> FloatVector<SIZE> {
        let mut ret_elements: [f32; SIZE] =
            unsafe { std::mem::MaybeUninit::uninit().assume_init() };
        for i in 0..SIZE {
            ret_elements[i] = self.elements[i] / other.into()
        }
        FloatVector::from_elements(ret_elements)
    }
}

impl<const SIZE: usize, T> DivAssign<T> for FloatVector<SIZE>
where
    T: Into<f32> + Copy,
{
    fn div_assign(&mut self, other: T) {
        for i in 0..SIZE {
            self.elements[i] /= other.into()
        }
    }
}

impl<const SIZE: usize> Index<usize> for FloatVector<SIZE> {
    type Output = f32;

    fn index(&self, index: usize) -> &f32 {
        &self.elements[index]
    }
}

impl<const SIZE: usize> IndexMut<usize> for FloatVector<SIZE> {
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.elements[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ops() {
        const SIZE_A: usize = 3;

        let mut aa: FloatVector<SIZE_A> = FloatVector::from_elements([1., 2., 3.]);
        let ab: FloatVector<SIZE_A> = FloatVector::from_elements([1., 1., 1.]);

        assert_eq!(aa.dot(&ab), 6.);
        assert_eq!(&aa + &ab, FloatVector::from_elements([2., 3., 4.]));
        aa += &ab;
        assert_eq!(aa, FloatVector::from_elements([2., 3., 4.]));
        assert_eq!(&aa - &ab, FloatVector::from_elements([1., 2., 3.]));
        aa -= &ab;
        assert_eq!(aa, FloatVector::from_elements([1., 2., 3.]));
    }

    #[test]
    fn test_iter_collect() {
        const SIZE_A: usize = 3;

        let aa: FloatVector<SIZE_A> = FloatVector::from_elements([1., 2., 3.]);
        let bb: FloatVector<SIZE_A> = aa.iter().map(|element| element * 2.0).collect();
        assert_eq!(bb, &aa * 2.0);
    }
}
