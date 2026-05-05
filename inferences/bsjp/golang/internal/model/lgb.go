package model

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

type Tree struct {
	nodes    []node
	leaves   []float64
}

type node struct {
	feature     int
	threshold   float64
	left, right int
	isLeaf      bool
	leafVal     float64
}

type LightGBM struct {
	trees       []*Tree
	featureMap  map[string]int // feature name → index
	nFeatures   int
}

func LoadLightGBM(path string) (*LightGBM, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	m := &LightGBM{}
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 1<<20), 1<<24) // large buffer for long lines

	var curTree *Tree
	var treeSizes []int
	parsingTree := false
	treeHead := make(map[string]string)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "feature_names=") {
			names := strings.Fields(line[len("feature_names="):])
			m.featureMap = make(map[string]int, len(names))
			for i, n := range names {
				m.featureMap[n] = i
			}
			m.nFeatures = len(names)
			continue
		}

		if strings.HasPrefix(line, "tree_sizes=") {
			for _, s := range strings.Split(line[len("tree_sizes="):], " ") {
				if s == "" {
					continue
				}
				v, _ := strconv.Atoi(s)
				treeSizes = append(treeSizes, v)
			}
			continue
		}

		if strings.HasPrefix(line, "Tree=") {
			if curTree != nil {
				_ = finalizeTree(curTree, treeHead)
				m.trees = append(m.trees, curTree)
			}
			curTree = &Tree{}
			treeHead = make(map[string]string)
			parsingTree = true
			continue
		}

		if parsingTree && strings.Contains(line, "=") {
			parts := strings.SplitN(line, "=", 2)
			key := strings.TrimSpace(parts[0])
			val := parts[1]
			treeHead[key] = val
		}
	}

	if curTree != nil {
		_ = finalizeTree(curTree, treeHead)
		m.trees = append(m.trees, curTree)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	fmt.Printf("Loaded %d trees, %d features\n", len(m.trees), m.nFeatures)
	return m, nil
}

func finalizeTree(t *Tree, head map[string]string) error {
	nLeaves, _ := strconv.Atoi(head["num_leaves"])
	if nLeaves == 0 {
		return fmt.Errorf("missing num_leaves")
	}

	splitFeatures := parseInts(head["split_feature"])
	thresholds := parseFloats(head["threshold"])
	leftChildren := parseInts(head["left_child"])
	rightChildren := parseInts(head["right_child"])
	leafValues := parseFloats(head["leaf_value"])

	nInternal := nLeaves - 1
	if len(splitFeatures) < nInternal || len(thresholds) < nInternal {
		return fmt.Errorf("not enough internal nodes")
	}

	t.nodes = make([]node, nInternal)
	t.leaves = make([]float64, nLeaves)
	copy(t.leaves, leafValues)

	for i := 0; i < nInternal; i++ {
		t.nodes[i].feature = splitFeatures[i]
		t.nodes[i].threshold = thresholds[i]
		t.nodes[i].left = leftChildren[i]
		t.nodes[i].right = rightChildren[i]
	}

	return nil
}

func (m *LightGBM) Predict(features map[string]float64) float64 {
	vec := make([]float64, m.nFeatures)
	for name, idx := range m.featureMap {
		if v, ok := features[name]; ok {
			vec[idx] = v
		}
	}
	return m.predictVec(vec)
}

// NumTrees returns the number of trees in the model.
func (m *LightGBM) NumTrees() int { return len(m.trees) }

// NumFeatures returns the number of features in the model.
func (m *LightGBM) NumFeatures() int { return m.nFeatures }

func (m *LightGBM) predictVec(vec []float64) float64 {
	var rawScore float64
	for _, t := range m.trees {
		rawScore += walkTree(t, vec)
	}
	return sigmoid(rawScore)
}

func walkTree(t *Tree, vec []float64) float64 {
	idx := 0 // start at root
	nInternal := len(t.nodes)
	for {
		if idx < 0 {
			// leaf: negative index means leaf (-1 → leaf 0, -10 → leaf 9)
			leafIdx := -idx - 1
			if leafIdx < len(t.leaves) {
				return t.leaves[leafIdx]
			}
			return 0
		}
		if idx >= nInternal {
			// out of range, shouldn't happen
			return 0
		}
		n := &t.nodes[idx]
		fVal := 0.0
		if n.feature < len(vec) {
			fVal = vec[n.feature]
		}
		if fVal <= n.threshold {
			idx = n.left
		} else {
			idx = n.right
		}
	}
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func parseInts(s string) []int {
	parts := strings.Fields(s)
	out := make([]int, len(parts))
	for i, p := range parts {
		out[i], _ = strconv.Atoi(p)
	}
	return out
}

func parseFloats(s string) []float64 {
	parts := strings.Fields(s)
	out := make([]float64, len(parts))
	for i, p := range parts {
		out[i], _ = strconv.ParseFloat(p, 64)
	}
	return out
}
