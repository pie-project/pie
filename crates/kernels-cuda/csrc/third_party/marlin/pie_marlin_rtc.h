#pragma once
// SPDX-License-Identifier: Apache-2.0
//
// The four standard-library names `scalar_type.hpp` needs when the compiler
// is NVRTC. Authored by pie; not vendored, not upstream, and deliberately
// named so — `new-horizon.md` §13.4 fixes the rule: impersonate a vendor
// header's spelling only when the includer is source we cannot edit, and
// `scalar_type.hpp` is a file this tree already modifies.
//
// # Why this file exists, and what breaks without it
//
// `vllm::ScalarType` encodes its six fields into one `int64_t` so that C++17
// can pass it as a template argument, and `marlin_template.h` decodes it back
// on the device side:
//
//     static constexpr auto b_type = vllm::ScalarType::from_id(b_type_id);
//
// Both directions are folds. `id()` accumulates through a
// `std::pair<Id, uint32_t>`; `from_id()` accumulates through a `std::tuple`
// grown one field at a time by `std::tuple_cat`, then hands it to the
// constructor with `std::apply`. Neither is host code — they are `constexpr`
// evaluations that happen the moment NVRTC resolves
// `Marlin<vllm::kU4B8.id(), …>`, which is the first thing a JIT compile of
// marlin does. So `#ifndef __CUDACC_RTC__` around `<tuple>` and `<utility>`
// would delete the directive and leave every use: the exact failure
// `csrc/vendor/type_traits`' own header records, one library over.
//
// NVRTC 13.0 answers no standard header at all — measured, 0 of 31
// (`new-horizon.md` §13.1) — so the names have to be carried. This carries
// **four**: `decay_t`, `pair`, `tuple` and the three free functions that
// operate on a tuple. Not a standard library; the closure of what one file
// asks for, which is the same discipline `csrc/vendor/type_traits` states for
// its eight traits. A fifth name is one entry and a compile error that says
// which.
//
// # Why the names are `std::`
//
// Because the bodies that use them are vendored and must stay byte-identical.
// The alternative — `pie::pair` — is a rename inside `id()`, `from_id()` and
// every structured binding over their results, which is a diff that reads as
// a rewrite rather than as a guard. `csrc/vendor/type_traits` and
// `csrc/vendor/cstdint` make the same trade for the same reason, and the
// blast radius is the same: this header is reachable only under
// `__CUDACC_RTC__`, where there is no other `namespace std` to collide with.
//
// # What is NOT here
//
// `std::variant`, `std::string`, `std::runtime_error`. Every use of those is
// inside a member `scalar_type.hpp` guards away for the device compile, so
// emulating them would be carrying a promise nothing asks of us — and a
// `variant` whose `visit` silently picked the wrong alternative is a wrong
// ANSWER rather than a missing name.

#include <cstdint>
#include <type_traits>

namespace std {

// ---------------------------------------------------------------------------
// <type_traits>, continued
// ---------------------------------------------------------------------------
//
// `csrc/vendor/type_traits` carries `is_same`, `is_same_v`, `conditional_t`,
// `void_t` and `declval`, which is what FlashInfer's closure asks for.
// `ScalarType::member_id_field_width` asks for one more: `decay_t`, applied
// to the `Member` a fold is currently looking at, which arrives as a
// reference. The two `remove_*` traits under it are what `decay_t` is defined
// in terms of, not extras.

/// `T&`, `T&&` -> `T`.
template <class T>
struct remove_reference {
  using type = T;
};
template <class T>
struct remove_reference<T&> {
  using type = T;
};
template <class T>
struct remove_reference<T&&> {
  using type = T;
};
template <class T>
using remove_reference_t = typename remove_reference<T>::type;

/// `const T`, `volatile T` -> `T`.
template <class T>
struct remove_cv {
  using type = T;
};
template <class T>
struct remove_cv<const T> {
  using type = T;
};
template <class T>
struct remove_cv<volatile T> {
  using type = T;
};
template <class T>
struct remove_cv<const volatile T> {
  using type = T;
};
template <class T>
using remove_cv_t = typename remove_cv<T>::type;

/// What a by-value parameter would deduce to.
///
/// The standard's `decay` also decays arrays to pointers and functions to
/// function pointers. `member_id_field_width` is applied to `uint8_t`, `bool`,
/// `int32_t` and an enum, so neither case can arise here and neither is
/// written: a trait that handles a case its only caller cannot produce is a
/// claim nothing checks.
template <class T>
using decay_t = remove_cv_t<remove_reference_t<T>>;

// ---------------------------------------------------------------------------
// <utility>
// ---------------------------------------------------------------------------

/// Two values, decomposable by a structured binding.
///
/// `first` and `second` are public members and there is no `tuple_size`
/// specialisation for this type, so `auto [id, bit_offset] = result;` binds
/// MEMBERWISE — which is the rule that lets a fifteen-line aggregate stand in
/// for the standard's `pair` without also carrying `tuple_element` and three
/// overloads of `get`.
template <class T1, class T2>
struct pair {
  T1 first{};
  T2 second{};

  constexpr pair() = default;
  // A CONSTRUCTOR TEMPLATE, and neither half of that is cosmetic. `id()`
  // returns `{id | …, bit_offset + bits}`, whose operands have promoted to
  // `uint64_t` and `unsigned long`; braced initialisation checks every
  // element for NARROWING against the parameter it binds to, and EDG makes
  // that an error rather than a warning as soon as the class stops being an
  // aggregate. Deducing the parameter types makes each binding an identity
  // conversion and puts the narrowing where the vendored code always meant it
  // — one explicit `static_cast` per field, into the width `Id` declares.
  template <class U1, class U2>
  __host__ __device__ constexpr pair(U1 a, U2 b)
      : first(static_cast<T1>(a)), second(static_cast<T2>(b)) {}
};

/// `{a, b}` with the types deduced, for the one call site that spells it.
template <class T1, class T2>
__host__ __device__ constexpr pair<decay_t<T1>, decay_t<T2>> make_pair(T1&& a,
                                                                       T2&& b) {
  return {static_cast<T1&&>(a), static_cast<T2&&>(b)};
}

/// `0, 1, … N-1` as a type, which is how `apply` reaches a tuple's elements.
template <class T, T... Is>
struct integer_sequence {};
template <size_t... Is>
using index_sequence = integer_sequence<size_t, Is...>;

namespace pie_detail {
template <size_t N, size_t... Is>
struct make_index : make_index<N - 1, N - 1, Is...> {};
template <size_t... Is>
struct make_index<0, Is...> {
  using type = index_sequence<Is...>;
};
}  // namespace pie_detail

template <size_t N>
using make_index_sequence = typename pie_detail::make_index<N>::type;

// ---------------------------------------------------------------------------
// <tuple>
// ---------------------------------------------------------------------------
//
// A head/tail list rather than a multiple-inheritance layout, because the only
// operations `from_id` performs on it are "append one element" and "expand
// into a constructor call", and a recursive list makes both fifteen lines of
// constexpr with no EBO, no `tuple_leaf` and no index arithmetic to get wrong.
// The layout is never observed: nothing takes `sizeof`, nothing casts, and the
// tuple never leaves the constant evaluation that built it.

template <class... Ts>
struct tuple;

template <>
struct tuple<> {};

template <class T, class... Ts>
struct tuple<T, Ts...> {
  T head{};
  tuple<Ts...> tail{};

  constexpr tuple() = default;
  __host__ __device__ constexpr tuple(T h, tuple<Ts...> t)
      : head(h), tail(t) {}
};

/// The empty tuple, so `make_tuple`'s recursion has a base case that is
/// DECLARED before the recursive form names it.
__host__ __device__ constexpr tuple<> make_tuple();

namespace pie_detail {
/// The `I`th element, by walking the tail `I` times.
template <size_t I>
struct tuple_at {
  template <class T, class... Ts>
  __host__ __device__ static constexpr auto get(const tuple<T, Ts...>& t) {
    return tuple_at<I - 1>::get(t.tail);
  }
};
template <>
struct tuple_at<0> {
  template <class T, class... Ts>
  __host__ __device__ static constexpr T get(const tuple<T, Ts...>& t) {
    return t.head;
  }
};

/// How many elements a tuple holds.
template <class T>
struct tuple_len;
template <class... Ts>
struct tuple_len<tuple<Ts...>> {
  static constexpr size_t value = sizeof...(Ts);
};

/// `f(get<Is>(t)...)`, with `Is` supplied by the sequence.
template <class F, class Tup, size_t... Is>
__host__ __device__ constexpr auto apply_at(F f, const Tup& t,
                                            index_sequence<Is...>) {
  return f(tuple_at<Is>::get(t)...);
}

}  // namespace pie_detail

/// `{a, b, …}` with the types deduced.
template <class T, class... Ts>
__host__ __device__ constexpr tuple<decay_t<T>, decay_t<Ts>...> make_tuple(
    T a, Ts... rest) {
  return tuple<decay_t<T>, decay_t<Ts>...>{a, make_tuple(rest...)};
}
__host__ __device__ constexpr tuple<> make_tuple() { return tuple<>{}; }
namespace pie_detail {
/// `tuple<As..., Bs...>`, built by prepending `a`'s elements one at a time.
///
/// Two arguments only, because `from_id` calls
/// `tuple_cat(accumulated, make_tuple(one_value))` and nothing else. A
/// variadic `tuple_cat` is the same recursion with a fold on top, and it is
/// left out for the reason the whole header is short.
template <class... Bs>
__host__ __device__ constexpr tuple<Bs...> cat2(const tuple<>&,
                                                const tuple<Bs...>& b) {
  return b;
}
template <class A, class... As, class... Bs>
__host__ __device__ constexpr tuple<A, As..., Bs...> cat2(
    const tuple<A, As...>& a, const tuple<Bs...>& b) {
  return tuple<A, As..., Bs...>{a.head, cat2(a.tail, b)};
}
}  // namespace pie_detail

/// Concatenate two tuples.
template <class... As, class... Bs>
__host__ __device__ constexpr auto tuple_cat(const tuple<As...>& a,
                                             const tuple<Bs...>& b) {
  return pie_detail::cat2(a, b);
}

/// `f(elements of t...)`.
template <class F, class... Ts>
__host__ __device__ constexpr auto apply(F f, const tuple<Ts...>& t) {
  return pie_detail::apply_at(f, t, make_index_sequence<sizeof...(Ts)>{});
}

}  // namespace std
