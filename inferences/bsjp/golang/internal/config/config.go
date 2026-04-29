package config

import (
	"fmt"
	"os"
	"path/filepath"
)

type Config struct {
	RepoRoot   string
	L0Dir      string
	L2Parquet  string
	DuckDBPath string
	ModelDir   string
}

func FromCWD() (*Config, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	root, err := findRepoRoot(cwd)
	if err != nil {
		return nil, err
	}
	return FromRoot(root), nil
}

func FromRoot(root string) *Config {
	return &Config{
		RepoRoot:   root,
		L0Dir:      filepath.Join(root, "data", "Level_0_Raw"),
		L2Parquet:  filepath.Join(root, "data", "Level_2_Datamart", "training_datamart_bsjp_overnight.parquet"),
		DuckDBPath: filepath.Join(root, "inferences", "bsjp", "db", "inference.duckdb"),
		ModelDir:   filepath.Join(root, "model", "BSJP"),
	}
}

func (c *Config) L0File(name string) string {
	return filepath.Join(c.L0Dir, name)
}

func findRepoRoot(start string) (string, error) {
	cur := start
	for {
		candidate := filepath.Join(cur, "data", "Level_0_Raw")
		if info, err := os.Stat(candidate); err == nil && info.IsDir() {
			return cur, nil
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			return "", fmt.Errorf("cannot find repo root (data/Level_0_Raw/ not found)")
		}
		cur = parent
	}
}
