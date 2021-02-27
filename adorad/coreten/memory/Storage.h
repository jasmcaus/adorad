#include <coreten/memory/intrusive_ptr.h> 


// The Storage class holds the intrusive pointer to the actual Storage of coreten::Tensor 
// One or more tensors may point to the same underlying storage (and thus we ensure that if one tensor is 
// destroyed, the other tensors will still be able to point to the storage (for this reason, we employ reference 
// counting

class Storage {

private:
	intrusive_ptr __data; // Holds the actual data
	
};                        
