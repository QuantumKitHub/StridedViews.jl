module StridedViewsAdaptExt

using Adapt
using StridedViews

Adapt.adapt_structure(to, x::StridedView) =
    StridedView(adapt(to, parent(x)), size(x), strides(x), x.offset, x.op)

end
