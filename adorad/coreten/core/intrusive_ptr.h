#pragma once 

#include "adorad/coreten/Checks.h"

#include <atomic>
#include <stdexcept> 


// coreten::intrusive_ptr is a faster alternative of std::shared_ptr 
// 
// This design is motivated, in part, by the boost::intrusive_ptr and the coreten::intrusive_ptr 
// 
// Like the name indicates, it's intrusive - reference counting is included directly in the managed class as opposed 
// to std::shared_ptr where the ref counter has to be dynamically allocated to live beside the object 
// Considering the memory footprint, intrusive_ptr is the same as that of a raw pointer (this, however, is not the
// same case for the std::shared_ptr that holds a pointer to the object, a pointer to the counter and the counter 
// itself). 
// 
// Consider a class X like:
//      class X {
//          std::string name;
//          int age;
//      };
// 
// Using this in your code with an std::shared_ptr:
//      
//      void test() {
//           std::shared_ptr<X> x(new X);
//           std::cout << x->name << std::endl;
//      }
// 
// To use an intrusive_ptr, you'd have to have a reference counter inside the class: 
//      class X {
//          std::string name;
//          int age;
//          long references; 
// 
//          X() : references(0) {}
//      };
// 
// and indicate to the intrusive_ptr where the reference counter can be found for this class:
//      inline void intrusive_ptr_addRef(X* x) 
//          { ++x->references; }
// 
//      inline void intrusive_ptr_delete(X* x)
//          { if(--x->references == 0) delete x; }
// 
// finally, you can use the intrusive_ptr to replace the std::shared_ptr:
//      
//      void test() {
//           coreten::intrusive_ptr<X> x(new X);
//           std::cout << x->name << std::endl;
//      }
// 
// As you can see, the pointer is very intrusive and needs some boilerplate code added to your application, 
// but it can leads to some interesting improvements for classes that are very often dynamically allocated.

// There is another advantage in using intrusive_ptr. As the reference counter is stored into the object itself,
// you can create several intrusive_ptr to the same object without any problem. This is not the case when you use 
// a shared_ptr. Indeed, if you create two shared_ptr to the same dynamically allocated object, they will both have 
// a different references counter and at the end, you will end up with an object being deleted twice.

// Of course, there are not only advantages. First of all, you have to declare a field in every classes that you want 
// to manage using an intrusive_ptr and you have to declare functions to manage the reference. Then there are some
// disadvantages when using this pointer type compared to a shared_ptr :

//       1. It's impossible to create a weak_ptr from a intrusive_ptr
//       2. Code redundancy -  you have to copy the reference counter in every class that you want to use an 
//          intrusive_ptr with
//       3. You have to provide a function for every types that has to be used with intrusive_ptr (only two functions 
//          if you use the template versions of the two functions)
// 
// To conclude, coreten::intrusive_ptr can be a good replacement of std::shared_ptr in a performance critical 
// application, but if you have no performances problem, do not use it because it makes your code less clear. If you 
// are concerned by performances when using std::shared_ptr, consider also using std::make_shared to create your 
// pointers, so that the reference counter and the object itself will be allocated at the same place and at the same 
// time, resulting in better performances. Another case where it's interesting to use an intrusive_ptr is when dealing 
// with libraries that use a lot of raw pointers, because you can create several intrusive_ptr to the same raw pointer
// without any problem.
// 
// 
// Source for this mini docstring: https://baptiste-wicht.com/posts/2011/11/boost-intrusive_ptr.html 
// Further reading on Boost's intrusive_ptr: http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p0468r1.html 


namespace coreten {
class intrusive_ptr_target;

namespace raw {
namespace weak_intrusive_ptr {
    inline void incref(intrusive_ptr_target* self);
} // namespace weak_intrusive_ptr

namespace intrusive_ptr {
    inline void incref(intrusive_ptr_target* self);
} //namespace (raw::)intrusive_ptr 

// constructor tag used by intrusive_ptr constructors
struct DontIncreaseRefcount {};

} // namespace raw 


// Similar to coreten::intrusive_ptr_target 
class intrusive_ptr_target {

    mutable std::atomic<size_t> __refcount;
    mutable std::atomic<size_t> __weakcount;

    template <typename T, typename NullType>
    friend class intrusive_ptr;
    friend inline void raw::intrusive_ptr::incref(intrusive_ptr_target* self);

    template <typename T, typename NullType>
    friend class weak_intrusive_ptr;
    friend inline void raw::weak_intrusive_ptr::incref(intrusive_ptr_target* self);

protected:
    // Protected Destructor 
    // Ideally, you'll never want to destruct intrusive_ptr* directly 
    virtual ~intrusive_ptr_target() {
        // Disable -Wterminate and -Wexceptions so we're allowed to use assertions
        // (i.e. throw exceptions) in a destructor.
        // We also have to disable -Wunknown-warning-option and -Wpragmas, because
        // some other compilers don't know about -Wterminate or -Wexceptions and
        // will show a warning about unknown warning options otherwise.
        #if defined(_MSC_VER) && !defined(__clang__)
            #pragma warning(push)
            #pragma warning(disable: 4297) // function assumed not to throw an exception but does
        #else
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Wpragmas"
            #pragma GCC diagnostic ignored "-Wunknown-warning-option"
            #pragma GCC diagnostic ignored "-Wterminate"
            #pragma GCC diagnostic ignored "-Wexceptions"
        #endif

        CORETEN_CHECK(
            __refcount.load() == 0,
            "Attempted to destruct an intrusive_ptr_target that still has an intrusive_ptr to it"
        );
        CORETEN_CHECK(
            __weakcount.load() == 0 || __weakcount.load() == 1,
            "Attempted to destruct an intrusive_ptr_target that still has a weak_intrusive_ptr to it"
        );

        #if defined(_MSC_VER) && !defined(__clang__)
            #pragma warning(pop)
        #else
            #pragma GCC diagnostic pop
        #endif
    }

    constexpr intrusive_ptr_target() noexcept : __refcount(0), __weakcount(0) {} 

    // intrusive_ptr_target supports only copy and move, but __refcount and __weakcount don't participate (since
    // they are intrinsic properties of the memory location)
    intrusive_ptr_target(intrusive_ptr_target&& other_ptr) noexcept: intrusive_ptr_target() {} 
    intrusive_ptr_target(const intrusive_ptr_target& other_ptr) noexcept: intrusive_ptr_target() {} 
    intrusive_ptr_target& operator=(intrusive_ptr_target&& other_ptr) noexcept { 
        return *this; 
    }
    intrusive_ptr_target& operator=(const intrusive_ptr_target& other_ptr) noexcept { 
        return *this; 
    }

private:
    // Called when __refcount reaches zero. 
    // This can be overridden to release expensive resources
    // There still might exist weak references, so your object might not get destructed just yet, but you can assume
    // that the object isn't used anymore, 
    // i.e no more calls to methods or accesses to member (you can destruct it yet because the __weakcount needs to be 
    // accessible). 
    // 
    // If there are no weak references, this function is guaranteed to be called first. 
    // However, if you use your class for an object on the stack that is desctucted by the scope (i.e without the 
    // intrusive_ptr), this function will not be called. 
    // 
    virtual void release_resources() {} 
}; // class intrusive_ptr_target


namespace detail {

template <class Target>
struct intrusive_target_default_null_type final {
    static constexpr Target* singleton() noexcept {
        return nullptr;
    }
};

template <class Target, class toNullType, class fromNullType>
Target* assign_ptr(Target* rhs) {
    if(fromNullType::singleton() == rhs) { 
        return toNullType::singleton(); 
    } else { 
        return rhs; 
    }
}

// Increment needs to be acquire-release to make use_count() and
// unique() reliable.
inline size_t atomic_refcount_increment(std::atomic<size_t>& refcount) {
    return refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
}

// weak_use_count() is only used for testing, so we don't need it to
// be reliable. Relaxed should be fine.
inline size_t atomic_weakcount_increment(std::atomic<size_t>& weakcount) {
    return weakcount.fetch_add(1, std::memory_order_relaxed) + 1;
}

// Both decrements need to be acquire-release for correctness. See
// e.g. std::shared_ptr implementation.
inline size_t atomic_refcount_decrement(std::atomic<size_t>& refcount) {
    return refcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}

inline size_t atomic_weakcount_decrement(std::atomic<size_t>& weakcount) {
    return weakcount.fetch_sub(1, std::memory_order_acq_rel) - 1;
}

} // namespace detail


template <class Target, class NullType>
class weak_intrusive_ptr;

template <class Target, class NullType = detail::intrusive_target_default_null_type<Target>>
class intrusive_ptr final {
private:
    //  the following static assert would be nice to have but it requires the target class T to be fully defined when 
    // intrusive_ptr<T> is instantiated. This is a problem for classes that contain pointers to themselves.
    //  static_assert(std::is_base_of<intrusive_ptr_target, Target>::value,
    //      "intrusive_ptr can only be used for classes that inherit from intrusive_ptr_target.");
    #ifndef _WIN32
        // This static_assert triggers on MSVC
        // error C2131: expression did not evaluate to a constant
        static_assert(
            NullType::singleton() == NullType::singleton(),
            "NullType must have a constexpr singleton() method");
    #endif
        static_assert(
            std::is_base_of<Target, typename std::remove_pointer<decltype(NullType::singleton())>::type>::value,
            "NullType::singleton() must return a element_type* pointer");

    Target* target_;

    template <class Target2, class NullType2>
    friend class intrusive_ptr;
    friend class weak_intrusive_ptr<Target, NullType>;

    void retain_() {
        if (target_ != NullType::singleton()) {
            size_t new_refcount = detail::atomic_refcount_increment(target_->refcount_);
            CORETEN_CHECK(
                new_refcount != 1,
                "intrusive_ptr: Cannot increase refcount after it reached zero."
            );
        }
    }

    void reset_() noexcept {
        if (target_ != NullType::singleton() && detail::atomic_refcount_decrement(target_->refcount_) == 0) {
            // justification for const_cast: release_resources is basically a destructor
            // and a destructor always mutates the object, even for const objects.
            const_cast<std::remove_const_t<Target>*>(target_)->release_resources();

            // See comment above about weakcount. As long as refcount>0, weakcount is one larger than the actual number 
            // of weak references. So we need to decrement it here.
            if (target_->weakcount_.load(std::memory_order_acquire) == 1 || detail::atomic_weakcount_decrement(target_->weakcount_) == 0) {
                delete target_;
            }
        }
        target_ = NullType::singleton();
    }

    // raw pointer constructors are not public because we shouldn't make intrusive_ptr out of raw pointers except 
    // from inside the make_intrusive(), reclaim() and weak_intrusive_ptr::lock() implementations.

    // This constructor will NOT increase the ref counter..
    // We use the tagged dispatch mechanism to explicitly mark this constructor to not increase the refcount
    explicit intrusive_ptr(Target* target, raw::DontIncreaseRefcount) noexcept : target_(target) {}

    // This constructor WILL increase the ref counter.
    // This constructor will be used by the make_intrusive() which wraps the intrusive_ptr holder around the raw pointer 
    // and incref correspondingly
    explicit intrusive_ptr(Target* target) : intrusive_ptr(target, raw::DontIncreaseRefcount{}) {
        if (target_ != NullType::singleton()) {
            // We can't use retain_(), because we also have to increase weakcount and because we allow raising these values 
            // from 0, which retain_() has an assertion against.
            detail::atomic_refcount_increment(target_->refcount_);
            detail::atomic_weakcount_increment(target_->weakcount_);
        }
    }

public:
    using element_type = Target;

    intrusive_ptr() noexcept : intrusive_ptr(NullType::singleton(), raw::DontIncreaseRefcount{}) {}

    intrusive_ptr(intrusive_ptr&& rhs) noexcept : target_(rhs.target_) { 
        rhs.target_ = NullType::singleton(); 
    }

    template <class From, class FromNullType>
    /* implicit */ intrusive_ptr(intrusive_ptr<From, FromNullType>&& rhs) noexcept
            : target_(detail::assign_ptr_<Target, NullType, FromNullType>(rhs.target_)) {
        static_assert(std::is_convertible<From*, Target*>::value,
            "Type mismatch. intrusive_ptr move constructor got pointer of wrong type.");
        rhs.target_ = FromNullType::singleton();
    }


    intrusive_ptr(const intrusive_ptr& rhs) : target_(rhs.target_) { 
        retain_();  
    }

    template <class From, class FromNullType>
    /* implicit */ intrusive_ptr(const intrusive_ptr<From, FromNullType>& rhs) 
        : target_(detail::assign_ptr_<Target, NullType, FromNullType>(rhs.target_)) {
        static_assert(
            std::is_convertible<From*, Target*>::value,
            "Type mismatch. intrusive_ptr copy constructor got pointer of wrong type.");
        retain_();
    }

    ~intrusive_ptr() noexcept {
        reset_();
    }

    Target* get() const noexcept { 
        return target_; 
    }


    operator bool() const noexcept { 
        return target_ != NullType::singleton(); 
    }

    void reset() noexcept { 
        reset_(); 
    }

    void swap(intrusive_ptr& rhs) noexcept {
        Target* tmp = target_;
        target_ = rhs.target_;
        rhs.target_ = tmp;
    }

    // We do a lot of null-pointer checks in our code, good to have this be cheap.
    bool defined() const noexcept { 
        return target_ != NullType::singleton(); 
    }

    size_t use_count() const noexcept {
        if (target_ == NullType::singleton()) { return 0; }

        return target_->refcount_.load(std::memory_order_acquire);
    }

    size_t weak_use_count() const noexcept {
        if (target_ == NullType::singleton()) { return 0; }

        return target_->weakcount_.load(std::memory_order_acquire);
    }

    bool unique() const noexcept { 
        return use_count() == 1; 
    }

    // Returns an owning (!) pointer to the underlying object and makes the intrusive_ptr instance invalid. That means the 
    // refcount is not decreased. You *must* put the returned pointer back into a intrusive_ptr using intrusive_ptr::reclaim
    // (ptr) to properly destruct it. This is helpful for C APIs.
    Target* release() noexcept {
        Target* result = target_;
        target_ = NullType::singleton();
        return result;
    }

    // Takes an owning pointer to Target* and creates an intrusive_ptr that takes over ownership. That means the refcount 
    // is not increased. 
    // This is the counter-part to intrusive_ptr::release() and the pointer passed in *must* have been created using 
    // intrusive_ptr::release().
    static intrusive_ptr reclaim(Target* owning_ptr) { 
        return intrusive_ptr(owning_ptr, raw::DontIncreaseRefcount{}); 
    }


    // Allocate a heap object with args and wrap it inside a intrusive_ptr and incref. This is a helper function to let 
    // make_intrusive() access private intrusive_ptr constructors.
    template <class... Args>
    static intrusive_ptr make(Args&&... args) {
        auto result = intrusive_ptr(new Target(std::forward<Args>(args)...), raw::DontIncreaseRefcount{});

        // We just created result.target_, so we know no other thread has access to it, so we know we needn't care about 
        // memory ordering.
        // (On x86_64, a store with memory_order_relaxed generates a plain old `mov`, whereas an atomic increment does a 
        // lock-prefixed `add`, which is much more expensive: https://godbolt.org/z/eKPzj8.)
        CORETEN_CHECK(
            result.target_->refcount_ == 0 && result.target_->weakcount_ == 0,
            "intrusive_ptr: Newly-created target had non-zero refcounts. Does its constructor do something strange like incref or create "
            "an intrusive_ptr from `this`?"
        );

        result.target_->refcount_.store(1, std::memory_order_relaxed);
        result.target_->weakcount_.store(1, std::memory_order_relaxed);

        return result;
    }

    // Turn a **non-owning raw pointer** to an intrusive_ptr.
    // This method is potentially dangerous (as it can mess up refcount).
    static intrusive_ptr unsafe_reclaim_from_nonowning(Target* raw_ptr) {
        // See Note [Stack allocated intrusive_ptr_target safety]
        CORETEN_CHECK(
            raw_ptr == NullType::singleton() || raw_ptr->refcount_.load() > 0,
            "intrusive_ptr can only reclaim pointers that are owned by someone"
        );

        auto ptr = reclaim(raw_ptr); // doesn't increase refcount
        ptr.retain_();
        return ptr;
    }

    // 
    // Operator-stuff 
    // 
    template <class From, class FromNullType>
    intrusive_ptr& operator=(intrusive_ptr<From, FromNullType>&& rhs) &  noexcept {
        static_assert(std::is_convertible<From*, Target*>::value,
            "Type mismatch. intrusive_ptr move assignment got pointer of wrong type.");

        intrusive_ptr tmp = std::move(rhs);
        swap(tmp);
        return *this;
    }

    intrusive_ptr& operator=(intrusive_ptr&& rhs) & noexcept { 
        return operator=<Target, NullType>(std::move(rhs)); 
    }
    intrusive_ptr& operator=(const intrusive_ptr& rhs) & noexcept { 
        return operator=<Target, NullType>(rhs); 
    }

    template <class From, class FromNullType>
    intrusive_ptr& operator=(const intrusive_ptr<From, NullType>& rhs) & {
        static_assert(std::is_convertible<From*, Target*>::value,
            "Type mismatch: intrusive_ptr copy assignment got pointer of wrong type.");
        intrusive_ptr tmp = rhs;
        swap(tmp);
        return *this;
    }

    Target& operator*() const noexcept { 
        return *target_; 
    }

    Target* operator->() const noexcept { 
        return target_; 
    }

}; // class intrusive_ptr (final)


template <
    class Target,
    class NullType = detail::intrusive_target_default_null_type<Target>,
    class... Args>
inline intrusive_ptr<Target, NullType> make_intrusive(Args&&... args) {
    return intrusive_ptr<Target, NullType>::make(std::forward<Args>(args)...);
}


template <class Target, class NullType>
inline void swap(intrusive_ptr<Target, NullType>& lhs, intrusive_ptr<Target, NullType>& rhs) noexcept {
    lhs.swap(rhs);
}

// To allow intrusive_ptr inside std::map or std::set, we need operator<
template <class Target1, class NullType1, class Target2, class NullType2>
inline bool operator<(const intrusive_ptr<Target1, NullType1>& lhs, const intrusive_ptr<Target2, NullType2>& rhs) noexcept {
    return lhs.get() < rhs.get();
}

template <class Target1, class NullType1, class Target2, class NullType2>
inline bool operator==(const intrusive_ptr<Target1, NullType1>& lhs, const intrusive_ptr<Target2, NullType2>& rhs) noexcept {
    return lhs.get() == rhs.get();
}

template <class Target1, class NullType1, class Target2, class NullType2>
inline bool operator!=(const intrusive_ptr<Target1, NullType1>& lhs, const intrusive_ptr<Target2, NullType2>& rhs) noexcept {
    return !operator==(lhs, rhs);
}

template <typename Target, class NullType = detail::intrusive_target_default_null_type<Target>>
class weak_intrusive_ptr final {
private:
    static_assert(std::is_base_of<intrusive_ptr_target, Target>::value,
        "intrusive_ptr can only be used for classes that inherit from intrusive_ptr_target.");

    #ifndef _WIN32
    // This static_assert triggers on MSVC
    //  error C2131: expression did not evaluate to a constant
    static_assert(
        NullType::singleton() == NullType::singleton(),
        "NullType must have a constexpr singleton() method");
    #endif
    static_assert(
        std::is_base_of<Target, typename std::remove_pointer<decltype(NullType::singleton())>::type>::value,
        "NullType::singleton() must return a element_type* pointer");

    Target* target_;

    template <class Target2, class NullType2>
    friend class weak_intrusive_ptr;

    void retain_() {
        if (target_ != NullType::singleton()) {
            size_t new_weakcount = detail::atomic_weakcount_increment(target_->weakcount_);
            CORETEN_CHECK(
                new_weakcount != 1,
                "weak_intrusive_ptr: Cannot increase weakcount after it reached zero."
            );
        }
    }

    void reset_() noexcept {
        if (target_ != NullType::singleton() && detail::atomic_weakcount_decrement(target_->weakcount_) == 0) {
            delete target_;
        }
        target_ = NullType::singleton();
    }

    constexpr explicit weak_intrusive_ptr(Target* target) : target_(target) {}


public:
    using element_type = Target;

    explicit weak_intrusive_ptr(const intrusive_ptr<Target, NullType>& ptr) : weak_intrusive_ptr(ptr.get()) {
        retain_();
    }

    weak_intrusive_ptr(weak_intrusive_ptr&& rhs) noexcept : target_(rhs.target_) {
        rhs.target_ = NullType::singleton();
    }

    template <class From, class FromNullType>
    /* implicit */ weak_intrusive_ptr(weak_intrusive_ptr<From, FromNullType>&& rhs) noexcept
      : target_(detail::assign_ptr_<Target, NullType, FromNullType>(rhs.target_)) {
        static_assert(
            std::is_convertible<From*, Target*>::value,
            "Type mismatch. weak_intrusive_ptr move constructor got pointer of wrong type.");
        rhs.target_ = FromNullType::singleton();
    }

    weak_intrusive_ptr(const weak_intrusive_ptr& rhs) : target_(rhs.target_) {
        retain_();
    }

    template <class From, class FromNullType>
    /* implicit */ weak_intrusive_ptr(const weak_intrusive_ptr<From, FromNullType>& rhs)
        : target_(detail::assign_ptr_<Target, NullType, FromNullType>(rhs.target_)) {
        static_assert(
            std::is_convertible<From*, Target*>::value,
            "Type mismatch. weak_intrusive_ptr copy constructor got pointer of wrong type.");
        retain_();
    }

    ~weak_intrusive_ptr() noexcept {
        reset_();
    }

    weak_intrusive_ptr& operator=(weak_intrusive_ptr&& rhs) & noexcept {
        return operator=<Target, NullType>(std::move(rhs));
    }

    template <class From, class FromNullType>
    weak_intrusive_ptr& operator=(weak_intrusive_ptr<From, FromNullType>&& rhs) & noexcept {
        static_assert(
            std::is_convertible<From*, Target*>::value,
            "Type mismatch. weak_intrusive_ptr move assignment got pointer of wrong type.");

        weak_intrusive_ptr tmp = std::move(rhs);
        swap(tmp);
        return *this;
    }

    weak_intrusive_ptr& operator=(const weak_intrusive_ptr& rhs) & noexcept {
        return operator=<Target, NullType>(rhs);
    }

    weak_intrusive_ptr& operator=(const intrusive_ptr<Target, NullType>& rhs) & noexcept {
        weak_intrusive_ptr tmp(rhs);
        swap(tmp);
        return *this;
    }

    template <class From, class FromNullType>
    weak_intrusive_ptr& operator=(const weak_intrusive_ptr<From, NullType>& rhs) & {
        static_assert(std::is_convertible<From*, Target*>::value,
            "Type mismatch. weak_intrusive_ptr copy assignment got pointer of wrong type.");

        weak_intrusive_ptr tmp = rhs;
        swap(tmp);
        return *this;
    }

    void reset() noexcept {
        reset_();
    }

    void swap(weak_intrusive_ptr& rhs) noexcept {
        Target* tmp = target_;
        target_ = rhs.target_;
        rhs.target_ = tmp;
    }

    // NB: This should ONLY be used by the std::hash implementation for weak_intrusive_ptr.  Another way you could do this // is friend std::hash<weak_intrusive_ptr>, but this triggers two bugs:
    //
    //  (1) It triggers an nvcc bug, where std::hash in a friend class declaration gets preprocessed into hash, which then 
    //      cannot actually be found.  The error in this case looks like:
    //
    //        error: no template named 'hash'; did you mean 'std::hash'?
    //
    //  (2) On OS X, std::hash is declared as a struct, not a class.This twings:
    //
    //        error: class 'hash' was previously declared as a struct [-Werror,-Wmismatched-tags]
    //
    // Both of these are work-aroundable, but on the whole, I decided it would be simpler and easier to make work if we 
    // just expose an unsafe getter for target_
    //
    Target* _unsafe_get_target() const noexcept {
        return target_;
    }

    size_t use_count() const noexcept {
        if (target_ == NullType::singleton()) { return 0; }
        return target_->refcount_.load(std::memory_order_acquire); // refcount, not weakcount!
    }

    size_t weak_use_count() const noexcept {
        if (target_ == NullType::singleton()) { return 0; }
        return target_->weakcount_.load(std::memory_order_acquire);
    }

    bool expired() const noexcept {
        return use_count() == 0;
    }

    intrusive_ptr<Target, NullType> lock() const noexcept {
        if (expired()) { 
            return intrusive_ptr<Target, NullType>(); 
        } 
        else {
            auto refcount = target_->refcount_.load(std::memory_order_seq_cst);
            do {
                if (refcount == 0) {
                    // Object already destructed, no strong references left anymore.
                    // Return nullptr.
                    return intrusive_ptr<Target, NullType>();
                }
            } while (!target_->refcount_.compare_exchange_weak(refcount, refcount + 1));

            return intrusive_ptr<Target, NullType>(target_, raw::DontIncreaseRefcount{});
        }
    }

    // Returns an owning (but still only weakly referenced) pointer to the underlying object and makes the // 
    // weak_intrusive_ptr instance invalid.
    // That means the weakcount is not decreased. You *must* put the returned pointer back into a weak_intrusive_ptr using
    // weak_intrusive_ptr::reclaim(ptr) to properly destruct it. This is helpful for C APIs.
    Target* release() noexcept {
        Target* result = target_;
        target_ = NullType::singleton();
        return result;
    }

    // Takes an owning (but must be weakly referenced) pointer to Target* and creates a weak_intrusive_ptr that takes over 
    // ownership. This means that the weakcount is not increased. 
    // This is the counter-part to weak_intrusive_ptr::release() and the pointer passed in *must* have been created using 
    // weak_intrusive_ptr::release().
    static weak_intrusive_ptr reclaim(Target* owning_weak_ptr) {
        // See Note [Stack allocated intrusive_ptr_target safety]
        // if refcount > 0, weakcount must be >1 for weak references to exist.
        // see weak counting explanation at top of this file.
        // if refcount == 0, weakcount only must be >0.
        CORETEN_CHECK(
            owning_weak_ptr == NullType::singleton() || owning_weak_ptr->weakcount_.load() > 1 ||
            (owning_weak_ptr->refcount_.load() == 0 && owning_weak_ptr->weakcount_.load() > 0),
            "weak_intrusive_ptr: Can only weak_intrusive_ptr::reclaim() owning pointers that were created using weak_intrusive_ptr::release()."
        );
        return weak_intrusive_ptr(owning_weak_ptr);
    }

    template <class Target1, class NullType1, class Target2, class NullType2>
    friend bool operator<(const weak_intrusive_ptr<Target1, NullType1>& lhs, const weak_intrusive_ptr<Target2, NullType2>& rhs) noexcept;

    template <class Target1, class NullType1, class Target2, class NullType2>
    friend bool operator==(const weak_intrusive_ptr<Target1, NullType1>& lhs, const weak_intrusive_ptr<Target2, NullType2>& rhs) noexcept;

}; // class weak_intrusive_ptr


template <class Target, class NullType>
inline void swap(weak_intrusive_ptr<Target, NullType>& lhs, weak_intrusive_ptr<Target, NullType>& rhs) noexcept {
    lhs.swap(rhs);
}

// To allow weak_intrusive_ptr inside std::map or std::set, we need operator<
template <class Target1, class NullType1, class Target2, class NullType2>
inline bool operator<(const weak_intrusive_ptr<Target1, NullType1>& lhs, const weak_intrusive_ptr<Target2, NullType2>& rhs) noexcept {
    return lhs.target_ < rhs.target_;
}

template <class Target1, class NullType1, class Target2, class NullType2>
inline bool operator==(const weak_intrusive_ptr<Target1, NullType1>& lhs, const weak_intrusive_ptr<Target2, NullType2>& rhs) noexcept {
    return lhs.target_ == rhs.target_;
}

template <class Target1, class NullType1, class Target2, class NullType2>
inline bool operator!=(const weak_intrusive_ptr<Target1, NullType1>& lhs, const weak_intrusive_ptr<Target2, NullType2>& rhs) noexcept {
    return !operator==(lhs, rhs);
}


// Alias for documentary purposes, to more easily distinguish weak raw intrusive pointers from intrusive pointers.
using weak_intrusive_ptr_target = intrusive_ptr_target;

// This namespace provides some methods for working with raw pointers that subclass intrusive_ptr_target.  They are not provided as methods on intrusive_ptr_target, because ideally you would not need these methods at all (use smart pointers), 
// but if you are dealing with legacy code that still needs to pass around raw pointers, you may find these quite useful.
//
// An important usage note: some functions are only valid if you have a strong raw pointer to the object, while others are 
// only valid if you have a weak raw pointer to the object.  ONLY call intrusive_ptr namespace functions on strong 
// pointers, and weak_intrusive_ptr namespace functions on weak pointers.  If you mix it up, you may get an assert failure.

namespace raw {

namespace intrusive_ptr {

// WARNING: Unlike the reclaim() API, it is NOT valid to pass NullType::singleton to this function
inline void incref(intrusive_ptr_target* self) {
    if (self) {
      detail::atomic_refcount_increment(self->__refcount);
    }
}

// WARNING: Unlike the reclaim() API, it is NOT valid to pass NullType::singleton to this function
inline void decref(intrusive_ptr_target* self) {
    // Let it die
    coreten::intrusive_ptr<intrusive_ptr_target>::reclaim(self);
    // NB: Caller still has 'self' pointer, but it's now invalid. 
    // If you want more safety, used the actual ccoreten::intrusive_ptr class
}


template <typename T>
inline T* make_weak(T* self) {
    // NB: 'this' is a strong pointer, but we return a weak pointer
    auto ptr = coreten::intrusive_ptr<T>::reclaim(self);
    coreten::weak_intrusive_ptr<T> wptr(ptr);
    ptr.release();
    return wptr.release();
}

inline size_t use_count(intrusive_ptr_target* self) {
    auto ptr = coreten::intrusive_ptr<intrusive_ptr_target>::reclaim(self);
    auto r = ptr.use_count();
    ptr.release();
    return r;
}

} // namespace intrusive_ptr_target


namespace weak_intrusive_ptr {

inline void incref(weak_intrusive_ptr_target* self) {
    detail::atomic_weakcount_increment(self->__weakcount);
}

inline void decref(weak_intrusive_ptr_target* self) {
    // Let it die
    coreten::weak_intrusive_ptr<intrusive_ptr_target>::reclaim(self);
    // NB: You still "have" the 'self' pointer, but it's now invalid.
    // If you want more safety, used the actual coreten::weak_intrusive_ptr class
}

template <typename T>
    inline T* lock(T* self) {
    auto wptr = coreten::weak_intrusive_ptr<T>::reclaim(self);
    auto ptr = wptr.lock();
    wptr.release();
    return ptr.release();
}

// This gives the STRONG refcount of a WEAK pointer
inline size_t use_count(weak_intrusive_ptr_target* self) {
    auto wptr = coreten::weak_intrusive_ptr<intrusive_ptr_target>::reclaim(self);
    auto r = wptr.use_count();
    wptr.release();
    return r;
}

} // namespace weak_intrusive_ptr_target
} // namespace raw

} // namespace coreten


namespace std {
// To allow intrusive_ptr and weak_intrusive_ptr inside std::unordered_map or std::unordered_set, we need std::hash
template <class Target, class NullType>
struct hash<coreten::intrusive_ptr<Target, NullType>> {
    size_t operator()(const coreten::intrusive_ptr<Target, NullType>& x) const {
        return std::hash<Target*>()(x.get());
    }
};

template <class Target, class NullType>
struct hash<coreten::weak_intrusive_ptr<Target, NullType>> {
    size_t operator()(const coreten::weak_intrusive_ptr<Target, NullType>& x) const {
        return std::hash<Target*>()(x._unsafe_get_target());
    }
};

} // namespace std