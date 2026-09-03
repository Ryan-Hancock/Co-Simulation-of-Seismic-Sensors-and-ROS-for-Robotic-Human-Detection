package fk

import "math/cmplx"

// A small complex 4x4 linear algebra kernel.
//
// gonum's complex matrix support stops short of a solve or an exponential, and
// the alternative — carrying real and imaginary parts through the real
// routines — doubles the dimension and obscures the structure. Four by four is
// small enough that direct code is clearer and faster than either.

type cmat [4][4]complex128

type cvec [4]complex128

func (a cmat) mul(b cmat) cmat {
	var out cmat
	for i := range 4 {
		for j := range 4 {
			var s complex128
			for k := range 4 {
				s += a[i][k] * b[k][j]
			}
			out[i][j] = s
		}
	}
	return out
}

func (a cmat) apply(v cvec) cvec {
	var out cvec
	for i := range 4 {
		var s complex128
		for k := range 4 {
			s += a[i][k] * v[k]
		}
		out[i] = s
	}
	return out
}

// column returns one column as a vector.
func (a cmat) column(j int) cvec {
	var v cvec
	for i := range 4 {
		v[i] = a[i][j]
	}
	return v
}

// solve solves a x = b by Gaussian elimination with partial pivoting.
//
// Partial pivoting is not optional here. The columns of the system built in
// this package are wave amplitudes whose magnitudes differ by whatever the
// layering makes them, and without pivoting the elimination divides by
// whichever happens to be smallest.
func (a cmat) solve(b cvec) (cvec, bool) {
	m := a
	x := b
	for col := range 4 {
		// Pivot on the largest remaining magnitude in this column.
		best, bestRow := 0.0, col
		for r := col; r < 4; r++ {
			if v := cmplx.Abs(m[r][col]); v > best {
				best, bestRow = v, r
			}
		}
		if best == 0 {
			return cvec{}, false
		}
		m[col], m[bestRow] = m[bestRow], m[col]
		x[col], x[bestRow] = x[bestRow], x[col]

		p := m[col][col]
		for r := col + 1; r < 4; r++ {
			f := m[r][col] / p
			if f == 0 {
				continue
			}
			for c := col; c < 4; c++ {
				m[r][c] -= f * m[col][c]
			}
			x[r] -= f * x[col]
		}
	}
	// Back substitution.
	var out cvec
	for i := 3; i >= 0; i-- {
		s := x[i]
		for j := i + 1; j < 4; j++ {
			s -= m[i][j] * out[j]
		}
		out[i] = s / m[i][i]
	}
	return out, true
}
