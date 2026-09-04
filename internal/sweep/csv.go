package sweep

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"
)

// ReadDesign parses a design CSV: a header of axis names, then one row of
// values per run.
func ReadDesign(r io.Reader) (Design, error) {
	rows, err := csv.NewReader(r).ReadAll()
	if err != nil {
		return Design{}, fmt.Errorf("sweep: reading design: %w", err)
	}
	if len(rows) < 2 {
		return Design{}, fmt.Errorf("sweep: a design needs a header and at least one row")
	}
	d := Design{Columns: rows[0]}
	for i, rec := range rows[1:] {
		vals := make([]float64, len(rec))
		for j, s := range rec {
			v, err := strconv.ParseFloat(s, 64)
			if err != nil {
				return Design{}, fmt.Errorf("sweep: row %d column %d: %w", i+1, j+1, err)
			}
			vals[j] = v
		}
		d.Rows = append(d.Rows, vals)
	}
	return d, d.Validate()
}

// WriteResults writes the design back alongside its metrics, so that one file
// carries both what was asked and what came of it.
//
// Both, rather than metrics alone in design order. A results file that only
// works if the reader still has the design it came from is a file that will one
// day be read against a different design, and the failure is silent.
func WriteResults(w io.Writer, d Design, m []Metrics, errs []error) error {
	cw := csv.NewWriter(w)
	defer cw.Flush()

	head := append([]string{}, d.Columns...)
	head = append(head, Metrics{}.Names()...)
	head = append(head, "error")
	if err := cw.Write(head); err != nil {
		return err
	}
	for i, row := range d.Rows {
		rec := make([]string, 0, len(head))
		for _, v := range row {
			rec = append(rec, strconv.FormatFloat(v, 'g', 10, 64))
		}
		for _, v := range m[i].Values() {
			rec = append(rec, strconv.FormatFloat(v, 'g', 10, 64))
		}
		msg := ""
		if errs[i] != nil {
			msg = errs[i].Error()
		}
		rec = append(rec, msg)
		if err := cw.Write(rec); err != nil {
			return err
		}
	}
	cw.Flush()
	return cw.Error()
}
