#!/bin/sh
# Replaces or inserts lines in a file. Refuses anything ambiguous.
#
#   sh tools/splice.sh <file> <first line> <last line> <replacement file>
#   sh tools/splice.sh <file> before  <pattern> <replacement file>   # insert before the match
#   sh tools/splice.sh <file> after   <pattern> <replacement file>   # insert after the match
#   sh tools/splice.sh <file> replace <pattern> <replacement file>   # replace the matched line
#   sh tools/splice.sh <file> between <first> <replacement file> <last>  # replace a range
#
# The line-number form exists because it is sometimes what you have. The **pattern forms are
# the ones to use**, because they resolve the line here, where a pattern that matches nothing
# or matches twice is an error - rather than in a shell one-liner, where
#
#   A=$(grep -n '<pattern>' f | cut -d: -f1)
#   { sed -n "1,$((A-1))p" f; cat new; sed -n "$A,\$p" f; } > tmp && cp tmp f
#
# silently produces garbage when the grep finds nothing: `$((A-1))` is -1, the first sed prints
# nothing, the second errors, and `cp` installs whatever landed in tmp. That has now cost
# build_all.bat twice (duplicated) and build_gui.bat and build_view.bat once each (truncated to
# the replacement alone). The line-number form catches the empty bound; only the pattern form
# removes the reason to hand-roll it. See docs/RISK.md S6.
set -e

f="$1"
[ -f "$f" ] || { echo "splice: no such file: $f" >&2; exit 1; }

case "$2" in
  between)
    # Replace from the line matching $3 through the line matching $5, inclusive.
    #
    # This exists because the pattern modes above address a single line, so replacing a *range*
    # meant resolving two line numbers by hand - the very thing the pattern modes were added to
    # stop. A splice that did exactly that took two extra lines off the top of a function,
    # removing `run_.n_events = n_events` from G4RunManager::BeamOn, and example B1 went on
    # transporting two million events and reporting nothing. See docs/RISK.md S10.
    #
    # Both ends must match exactly once, and the second must not precede the first.
    first="$3"; last="$5"; new="$4"
    [ -f "$new" ] || { echo "splice: no such replacement: $new" >&2; exit 1; }
    [ -n "$last" ] || { echo "splice: 'between' needs <first> <newfile> <last>" >&2; exit 1; }
    for p in "$first" "$last"; do
      hits=$(grep -c -F -x -- "$p" "$f" || true)
      case "$hits" in
        0) echo "splice: pattern not found in $f:" >&2; echo "  $p" >&2; exit 1;;
        1) ;;
        *) echo "splice: pattern matches $hits lines in $f - not unique:" >&2
           echo "  $p" >&2
           grep -n -F -x -- "$p" "$f" >&2
           exit 1;;
      esac
    done
    a=$(grep -n -F -x -- "$first" "$f" | cut -d: -f1)
    b=$(grep -n -F -x -- "$last" "$f" | cut -d: -f1)
    if [ "$b" -lt "$a" ]; then
      echo "splice: the last pattern (line $b) is before the first (line $a)" >&2
      exit 1
    fi
    ;;
  before|after|replace)
    mode="$2"; pattern="$3"; new="$4"
    [ -f "$new" ] || { echo "splice: no such replacement: $new" >&2; exit 1; }
    hits=$(grep -c -F -x -- "$pattern" "$f" || true)
    case "$hits" in
      0) echo "splice: pattern not found in $f:" >&2; echo "  $pattern" >&2; exit 1;;
      1) ;;
      *) echo "splice: pattern matches $hits lines in $f - not unique:" >&2
         echo "  $pattern" >&2
         grep -n -F -x -- "$pattern" "$f" >&2
         exit 1;;
    esac
    ln=$(grep -n -F -x -- "$pattern" "$f" | cut -d: -f1)
    case "$mode" in
      before)  a="$ln"; b=$((ln - 1));;   # b < a means "insert, replace nothing"
      after)   a=$((ln + 1)); b="$ln";;
      replace) a="$ln"; b="$ln";;
    esac
    ;;
  *)
    a="$2"; b="$3"; new="$4"
    [ -f "$new" ] || { echo "splice: no such replacement: $new" >&2; exit 1; }
    case "$a" in ''|*[!0-9]*) echo "splice: first line '$a' is not a number" >&2; exit 1;; esac
    case "$b" in ''|*[!0-9]*) echo "splice: last line '$b' is not a number" >&2; exit 1;; esac
    ;;
esac

n=$(wc -l < "$f")
[ "$a" -ge 1 ] || { echo "splice: first line $a < 1" >&2; exit 1; }
[ "$b" -le "$n" ] || { echo "splice: last line $b past end of file ($n lines)" >&2; exit 1; }
if [ "$b" -lt "$((a - 1))" ]; then
  echo "splice: last line $b is more than one before first line $a" >&2
  exit 1
fi

tmp="$f.splice.$$"
{ [ "$a" -gt 1 ] && sed -n "1,$((a-1))p" "$f"
  cat "$new"
  [ "$b" -lt "$n" ] && sed -n "$((b+1)),\$p" "$f"
  true
} > "$tmp"
mv "$tmp" "$f"
if [ "$b" -lt "$a" ]; then
  echo "splice: $f inserted $(wc -l < "$new") lines at $a, now $(wc -l < "$f") lines"
else
  echo "splice: $f lines $a..$b -> $(wc -l < "$new") lines, now $(wc -l < "$f") lines"
fi
