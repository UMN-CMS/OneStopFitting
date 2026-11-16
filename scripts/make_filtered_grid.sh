(
src=filtered_signals
cd "$src"
mkdir -p ../subset
find . -type d -name '*_[0-9]*_[0-9]*_official' | sed 's|^\./||' | awk -F/ '
{
    n = split($NF, f, "_")
    x = f[n-2]; y = f[n-1]
    if (x !~ /^[0-9]+$/ || y !~ /^[0-9]+$/) next
    if (x%100==0 && y%100==0) keep=((x/100 + y/100) % 2 == 0) # interior: checkerboard
    else keep=(x%100==0) # diagonal edge: every other point
    if (keep) print $0
}' | xargs -r -d '\n' cp -r --parents -t ../subset/
)
