use any_intern::UnsafeLock;
use std::{cell::Cell, rc::Rc, thread};

fn main() {
    test_unsafelock();
}

#[allow(unused_mut)]
fn test_unsafelock() {
    let Ok(num_cpus) = thread::available_parallelism() else {
        return;
    };
    let mut num_threads = num_cpus.get();

    let val = Rc::new(Cell::new(0_usize));

    // UnsafeLock requires that there's no copies of the value.
    // We're cloning the value here, but we're not going to use the original value so it's fine.
    let lock = unsafe { UnsafeLock::new(val.clone()) };

    // Threads will increase the value. The value will become N * num_threads.
    const N: usize = 10_000;
    let mut handles = Vec::new();
    for _ in 0..num_threads {
        let c_lock = lock.clone();
        let handle = thread::spawn(move || {
            for _ in 0..N {
                unsafe {
                    let val = c_lock.lock().as_mut();
                    val.set(val.get() + 1);
                    c_lock.unlock();
                }
            }
        });
        handles.push(handle);
    }

    // UnsafeLock's guarantee is upheld when all data is under its protection. But, this block of
    // code violate the safety. You can check this through thread sanitizer.
    //
    // num_threads += 1;
    // for _ in 0..N {
    //     val.set(val.get() + 1); // val is outside the UnsafeLock
    // }

    for handle in handles {
        handle.join().unwrap();
    }

    unsafe {
        let val = lock.lock().as_ref();
        assert_eq!(val.get(), N * num_threads);
        lock.unlock();
    }
}
